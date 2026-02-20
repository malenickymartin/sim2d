import torch
from torch_geometric.data import HeteroData

from pathlib import Path
from tqdm import tqdm
from abc import ABC
from abc import abstractmethod
from typing import Optional
from collections import defaultdict

from .engine import EulerSolver
from .collisions import compute_collision
from .shapes import Floor
from .shapes import Shape
from .logger import EngineLogger, LoggingConfig
from .joints import Joint
from .joints import compute_joint_constraints
from .joints import JOINT_NUM_CONSTR
from .joints import JOINT_INT_TO_STR
from .joints import joint_to_int
from .gnn.dataset import NODE_FEATURE_DIMS
from .gnn.dataset import EDGE_FEATURE_DIMS


class Simulator(ABC):
    def __init__(
        self,
        sim_time,
        newton_iters: int = 50,
        gravity: torch.Tensor = torch.tensor([0.0, -9.81, 0.0]),
        dt: float = 0.01,
        init_gnn_path: Optional[str | Path] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )
        self.init_gnn_path = init_gnn_path

        self.num_steps = int(round(sim_time / dt))
        self.newton_iters = newton_iters
        self.gravity = gravity.to(self.device)
        self.dt = dt
        self.shapes: list[Shape] = []
        self.joints: list[Joint] = []
        self.ignore_contacts: list[tuple[int]] = []
        self.floor = None

        if logging_config is None:
            logging_config = LoggingConfig()
        self.logger = EngineLogger(logging_config)

        self.build_model()
        self.num_shapes = len(self.shapes)
        assert self.num_shapes > 0, "Cannot simulate nothing"
        assert not Floor in [type(s) for s in self.shapes], "Floor should be saved in self.floor"
        for shape in self.shapes:
            shape.to(self.device)
        for joint in self.joints:
            joint.to(self.device)

        self.gnn = None
        if not self.init_gnn_path is None:
            self.gnn = torch.load(self.init_gnn_path, self.device, weights_only=False)
            self.gnn.eval()

        self.solver = EulerSolver(
            self.shapes,
            self.newton_iters,
            self.gravity,
            self.dt,
            self.init_state_fn,
            self.logger,
            self.device,
        )

    def run(self):
        self.logger.open()
        self.logger.log_init_config(self)

        state = torch.zeros((self.num_shapes, 3), device=self.device)
        for i in range(self.num_shapes):
            state[i, :] = torch.cat(
                [self.shapes[i].velocity, self.shapes[i].angular_velocity.unsqueeze(0)]
            )
        with torch.no_grad():
            for i in tqdm(range(self.num_steps), desc="Simulation"):
                current_time = i * self.dt
                with self.logger.timed_block("contacts_and_joints"):
                    contacts_dict, contact_log = self.collide()
                    joints_dict, joint_log = self.process_joints()
                    all_constraints = self.merge_constraints(contacts_dict, joints_dict)
                self.logger.log_step_data(
                    i, current_time, self.shapes, state, contact_log, joint_log
                )
                with self.logger.timed_block("physics_step"):
                    state = self.solver.step(i, state, all_constraints)
                with self.logger.timed_block("update_shapes"):
                    self.update_shapes(state)
        _, contact_log = self.collide()
        _, joint_log = self.process_joints()
        self.logger.log_step_data(
            i + 1, current_time + self.dt, self.shapes, state, contact_log, joint_log
        )
        self.logger.close()

    def update_shapes(self, state):
        for i in range(self.num_shapes):
            self.shapes[i].translation += state[i][:2] * self.dt
            self.shapes[i].rotation = (self.shapes[i].rotation + state[i][2] * self.dt) % (
                2 * torch.pi
            )
            self.shapes[i].velocity = state[i][:2]
            self.shapes[i].angular_velocity = state[i][2]

    def collide(self) -> tuple[dict, dict]:
        contact_log = {"count": 0, "indices": [], "distances": [], "Js": []}
        c_body_idx = []
        c_local_idx = []
        c_dist = []
        c_jac = []
        c_jac_neigh = []
        c_neigh = []
        c_restitution = []
        c_counts = torch.zeros(self.num_shapes, dtype=torch.long, device=self.device)
        shapes = self.shapes + [self.floor] if not self.floor is None else self.shapes
        if len(shapes) >= 2:
            for i, shape_1 in enumerate(shapes):
                for j, shape_2 in enumerate(shapes[i + 1 :], i + 1):
                    if (i, j) in self.ignore_contacts or (j, i) in self.ignore_contacts:
                        continue
                    in_collision, distance, J_1, J_2 = compute_collision(
                        shape_1, shape_2, self.device
                    )
                    if in_collision:
                        i_2 = j if not isinstance(shape_2, Floor) else -1
                        if self.logger.config.enable_hdf5:
                            contact_log["count"] += 1
                            contact_log["distances"].append(float(distance))
                            contact_log["Js"].append((J_1.cpu(), J_2.cpu()))
                            contact_log["indices"].append((i, i_2))
                        c_body_idx.append(i)
                        c_neigh.append(i_2)
                        c_local_idx.append(c_counts[i].item())
                        c_dist.append(distance)
                        c_jac.append(J_1)
                        c_jac_neigh.append(J_2)
                        c_counts[i] += 1
                        restitutions = (
                            shape_1.restitution
                            + (
                                shape_2.restitution
                                if not isinstance(shape_2, Floor)
                                else self.floor.restitution
                            )
                        ) / 2
                        c_restitution.append(restitutions)

        contacts = {
            "body_idx": torch.tensor(c_body_idx, dtype=torch.long, device=self.device),
            "neighbor_idx": torch.tensor(c_neigh, dtype=torch.long, device=self.device),
            "local_idx": torch.tensor(c_local_idx, dtype=torch.long, device=self.device),
            "dist": torch.tensor(c_dist, dtype=torch.float32, device=self.device),
            "restitution": torch.tensor(c_restitution, dtype=torch.float32, device=self.device),
            "jac": (
                torch.stack(c_jac).to(self.device)
                if len(c_jac) > 0
                else torch.empty((0, 3), dtype=torch.float32, device=self.device)
            ),
            "jac_neigh": (
                torch.stack(c_jac_neigh).to(self.device)
                if len(c_jac_neigh) > 0
                else torch.empty((0, 3), dtype=torch.float32, device=self.device)
            ),
            "counts": c_counts,
            "is_equality": torch.zeros(len(c_body_idx), dtype=torch.bool, device=self.device),
        }
        return contacts, contact_log

    def process_joints(self):
        joint_log = {"count": 0, "indices": [], "error": [], "Js": []}
        c_body_idx, c_neigh, c_jac, c_jac_neigh, c_error = [], [], [], [], []
        for joint in self.joints:
            shape_1 = self.shapes[joint.child_idx]
            shape_2 = self.shapes[joint.parent_idx] if joint.parent_idx != -1 else None
            constrs = compute_joint_constraints(joint, shape_1, shape_2, self.device)
            for J_1, J_2, error in constrs:
                if self.logger.config.enable_hdf5:
                    joint_log["count"] += 1
                    joint_log["error"].append(float(error))
                    joint_log["Js"].append((J_1.cpu(), J_2.cpu()))
                    joint_log["indices"].append((joint.child_idx, joint.parent_idx))
                c_body_idx.append(joint.child_idx)
                c_neigh.append(joint.parent_idx)
                c_jac.append(J_1)
                c_jac_neigh.append(J_2)
                c_error.append(error)
        joints = {
            "body_idx": torch.tensor(c_body_idx, dtype=torch.long, device=self.device),
            "neighbor_idx": torch.tensor(c_neigh, dtype=torch.long, device=self.device),
            "error": torch.tensor(c_error, dtype=torch.float32, device=self.device),
            "jac": (
                torch.stack(c_jac).to(self.device)
                if len(c_jac) > 0
                else torch.empty((0, 3), dtype=torch.float32, device=self.device)
            ),
            "jac_neigh": (
                torch.stack(c_jac_neigh).to(self.device)
                if len(c_jac_neigh) > 0
                else torch.empty((0, 3), dtype=torch.float32, device=self.device)
            ),
        }
        return joints, joint_log

    def merge_constraints(self, contacts, joints):
        if not len(joints["body_idx"]) > 0:
            return contacts

        counts = contacts["counts"].clone()
        joints_local_idx = []
        for b in joints["body_idx"]:
            joints_local_idx.append(counts[b].item())
            counts[b] += 1

        merged = {
            "body_idx": torch.cat([contacts["body_idx"], joints["body_idx"]]),
            "neighbor_idx": torch.cat([contacts["neighbor_idx"], joints["neighbor_idx"]]),
            "local_idx": torch.cat(
                [
                    contacts["local_idx"],
                    torch.tensor(joints_local_idx, dtype=torch.long, device=self.device),
                ]
            ),
            "jac": torch.cat([contacts["jac"], joints["jac"]]),
            "jac_neigh": torch.cat([contacts["jac_neigh"], joints["jac_neigh"]]),
            "dist": torch.cat([contacts["dist"], joints["error"]]),
            "restitution": torch.cat(
                [
                    contacts["restitution"],
                    torch.zeros(len(joints["body_idx"]), dtype=torch.float32, device=self.device),
                ]
            ),
            "is_equality": torch.cat(
                [
                    contacts["is_equality"],
                    torch.ones_like(joints["body_idx"], dtype=torch.bool, device=self.device),
                ]
            ),
            "counts": counts,
        }
        return merged

    def create_gnn_data(self, state: torch.Tensor, constraints: torch.Tensor):
        gnn_data = HeteroData()
        gnn_data["object"].x = torch.zeros(
            (self.num_shapes, NODE_FEATURE_DIMS["object"]), dtype=torch.float32, device=self.device
        )
        for i in range(self.num_shapes):
            gnn_data["object"].x[i][:] = torch.tensor(
                [
                    self.shapes[i].mass,
                    self.shapes[i].inertia,
                    state[i][0],
                    state[i][1],
                    torch.norm(state[i][:2]),
                    state[i][2],
                ],
                device=self.device,
            )

        if (
            self.floor is not None
            or (constraints["neighbor_idx"][constraints["is_equality"]] == -1).any()
        ):
            gnn_data["floor"].x = torch.zeros((1, 0), dtype=torch.float32, device=self.device)
        else:
            gnn_data["floor"].x = torch.zeros((0, 0), dtype=torch.float32, device=self.device)

        mask_eq = constraints["is_equality"]

        # Contacts
        mask_contacts = ~mask_eq
        if mask_contacts.any():
            c_body = constraints["body_idx"][mask_contacts]
            c_neigh = constraints["neighbor_idx"][mask_contacts]
            c_attr = torch.cat(
                [
                    constraints["jac"][mask_contacts],
                    constraints["dist"][mask_contacts].unsqueeze(1),
                    constraints["restitution"][mask_contacts].unsqueeze(1),
                ],
                dim=1,
            )
            c_attr_neigh = torch.cat(
                [
                    constraints["jac_neigh"][mask_contacts],
                    constraints["dist"][mask_contacts].unsqueeze(1),
                    constraints["restitution"][mask_contacts].unsqueeze(1),
                ],
                dim=1,
            )
            mask_floor = c_neigh == -1
            if mask_floor.any():
                gnn_data["floor", "contact", "object"].edge_index = torch.stack(
                    [torch.zeros_like(c_body[mask_floor]), c_body[mask_floor]], dim=0
                )
                gnn_data["floor", "contact", "object"].edge_attr = c_attr[mask_floor]
            mask_obj = c_neigh != -1
            if mask_obj.any():
                src, dst = c_neigh[mask_obj], c_body[mask_obj]
                gnn_data["object", "contact", "object"].edge_index = torch.cat(
                    [torch.stack([src, dst], dim=0), torch.stack([dst, src], dim=0)], dim=1
                )
                gnn_data["object", "contact", "object"].edge_attr = torch.cat(
                    [c_attr[mask_obj], c_attr_neigh[mask_obj]], dim=0
                )

        # Joints
        if mask_eq.any():
            j_jac = constraints["jac"][mask_eq]
            j_jac_n = constraints["jac_neigh"][mask_eq]
            j_dist = constraints["dist"][mask_eq]

            curr = 0
            joint_edges = defaultdict(list)
            for joint in self.joints:
                n_c = JOINT_NUM_CONSTR[joint_to_int(joint)]
                j_name = JOINT_INT_TO_STR[joint_to_int(joint)]
                attr_child = torch.cat(
                    [
                        torch.cat([j_jac[curr + k], j_dist[curr + k : curr + k + 1]])
                        for k in range(n_c)
                    ]
                )
                if joint.parent_idx == -1:
                    joint_edges[(j_name, "floor")].append((0, joint.child_idx, attr_child))
                else:
                    attr_parent = torch.cat(
                        [
                            torch.cat([j_jac_n[curr + k], j_dist[curr + k : curr + k + 1]])
                            for k in range(n_c)
                        ]
                    )
                    joint_edges[(j_name, "object")].append(
                        (joint.parent_idx, joint.child_idx, attr_child)
                    )
                    joint_edges[(j_name, "object")].append(
                        (joint.child_idx, joint.parent_idx, attr_parent)
                    )
                curr += n_c

            for (j_name, src_type), edges in joint_edges.items():
                srcs, dsts, attrs = zip(*edges)
                edge_type = (src_type, j_name, "object")
                gnn_data[edge_type].edge_index = torch.tensor(
                    [srcs, dsts], dtype=torch.long, device=self.device
                )
                gnn_data[edge_type].edge_attr = torch.stack(attrs)

        self._ensure_empty_edges(gnn_data)
        return gnn_data

    def _ensure_empty_edges(self, data):
        for edge_type_key in EDGE_FEATURE_DIMS.keys():
            if edge_type_key not in data.edge_types:
                data[edge_type_key].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long, device=self.device
                )
                data[edge_type_key].edge_attr = torch.zeros(
                    (0, EDGE_FEATURE_DIMS[edge_type_key]), dtype=torch.float32, device=self.device
                )

    def state_from_gnn(self, gnn_output: tuple, constraints: dict) -> torch.Tensor:
        object_states, lambdas_dict = gnn_output
        state_guess = torch.zeros(self.solver.state_shape(constraints), device=self.device)
        state_guess[:, :3] = object_states

        mask_eq = constraints["is_equality"]
        mask_contacts = ~mask_eq

        c_body = constraints["body_idx"][mask_contacts]
        c_neigh = constraints["neighbor_idx"][mask_contacts]
        c_local = constraints["local_idx"][mask_contacts]

        mask_floor = c_neigh == -1
        mask_obj = ~mask_floor
        if mask_floor.any():
            body_idxs = c_body[mask_floor]
            local_idxs = c_local[mask_floor]
            pred_lambdas = lambdas_dict.get(("floor", "contact", "object"), torch.empty(0))
            if pred_lambdas.numel() > 0:
                n_floor = body_idxs.shape[0]
                state_guess[body_idxs, 3 + local_idxs] = pred_lambdas.view(-1)[:n_floor]
        if mask_obj.any():
            body_idxs = c_body[mask_obj]
            local_idxs = c_local[mask_obj]

            pred_lambdas = lambdas_dict.get(("object", "contact", "object"), torch.empty(0))
            if pred_lambdas.numel() > 0:
                n_obj = body_idxs.shape[0]
                state_guess[body_idxs, 3 + local_idxs] = pred_lambdas.view(-1)[:n_obj]

        joint_output_counters = defaultdict(int)

        j_body_all = constraints["body_idx"][mask_eq]
        j_local_all = constraints["local_idx"][mask_eq]

        joint_constraint_offset = 0
        for joint in self.joints:
            j_type_int = joint_to_int(joint)
            j_name = JOINT_INT_TO_STR[j_type_int]
            num_rows = JOINT_NUM_CONSTR[j_type_int]

            body_idx_for_joint = j_body_all[
                joint_constraint_offset : joint_constraint_offset + num_rows
            ]
            local_idxs_for_joint = j_local_all[
                joint_constraint_offset : joint_constraint_offset + num_rows
            ]
            body_idx = body_idx_for_joint[0]

            edge_key = (
                ("object", j_name, "object")
                if joint.parent_idx != -1
                else ("floor", j_name, "object")
            )
            if edge_key in lambdas_dict:
                preds = lambdas_dict[edge_key]
                idx = joint_output_counters[edge_key]
                if idx < preds.shape[0]:
                    val = preds[idx]
                    state_guess[body_idx, 3 + local_idxs_for_joint] = val
                    if joint.parent_idx != -1:
                        joint_output_counters[edge_key] += 2
                    else:
                        joint_output_counters[edge_key] += 1
            joint_constraint_offset += num_rows
        return state_guess

    def init_state_fn(self, state: torch.Tensor, constraints: torch.Tensor, dt: float):
        """
        return guess for next state of shape (self.num_shapes x 3+max([len(a) for a in contacts]))
        """
        if self.gnn is None:
            state_guess = torch.zeros(self.solver.state_shape(constraints), device=self.device)
            state_guess[:, :3] += state[:, :3]
            state_guess[:, :3] += dt * self.gravity
        else:
            gnn_data = self.create_gnn_data(state, constraints)
            gnn_output = self.gnn(
                gnn_data.x_dict, gnn_data.edge_index_dict, gnn_data.edge_attr_dict
            )
            state_guess = self.state_from_gnn(gnn_output, constraints)

        return state_guess

    @abstractmethod
    def build_model(self):
        """
        Fill the self.shapes list with instances of classes from src.shapes, fill the self.joints
        list with instances of classes from src.joints, and set self.floor
        """
        pass
