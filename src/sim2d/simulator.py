import torch
from torch_geometric.data import HeteroData

from pathlib import Path
from tqdm import tqdm
from abc import ABC
from abc import abstractmethod
from typing import Optional

from .engine import EulerSolver
from .collisions import compute_collision
from .shapes import Floor
from .shapes import Shape
from .logger import EngineLogger, LoggingConfig
from .joints import Joint, compute_joint_constraints


class Simulator(ABC):
    def __init__(
        self,
        sim_time,
        newton_iters: int = 50,
        gravity: torch.Tensor = torch.tensor([0.0, -9.81, 0.0]),
        dt: float = 0.01,
        init_gnn_path: Optional[str | Path] = None,
        logging_config: Optional[LoggingConfig] = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_steps = int(sim_time // dt + sim_time % dt)
        self.newton_iters = newton_iters
        self.gravity = gravity.to(self.device)
        self.dt = dt
        self.shapes: list[Shape] = []
        self.joints: list[Joint] = []
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
        if not init_gnn_path is None:
            self.gnn = torch.load(init_gnn_path, self.device, weights_only=False)
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
            self.shapes[i].rotation += state[i][2] * self.dt
            self.shapes[i].velocity = state[i][:2]
            self.shapes[i].angular_velocity = state[i][2]

    def collide(self) -> tuple[dict, dict]:
        contact_log = {"count": 0, "indices": [], "distances": [], "Js": []}
        c_body_idx = []
        c_local_idx = []
        c_dist = []
        c_jac = []
        c_neigh = []
        c_counts = torch.zeros(self.num_shapes, dtype=torch.long, device=self.device)
        shapes = self.shapes + [self.floor] if not self.floor is None else self.shapes
        if len(shapes) >= 2:
            for i, shape_1 in enumerate(shapes):
                for j, shape_2 in enumerate(shapes[i + 1 :], i + 1):
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
                        c_counts[i] += 1
        contacts = {
            "body_idx": torch.tensor(c_body_idx, dtype=torch.long, device=self.device),
            "neighbor_idx": torch.tensor(c_neigh, dtype=torch.long, device=self.device),
            "local_idx": torch.tensor(c_local_idx, dtype=torch.long, device=self.device),
            "dist": torch.tensor(c_dist, dtype=torch.float32, device=self.device),
            "jac": (
                torch.stack(c_jac).to(self.device)
                if len(c_jac) > 0
                else torch.empty((0, 3), dtype=torch.float32, device=self.device)
            ),
            "counts": c_counts,
            "is_equality": torch.zeros(len(c_body_idx), dtype=torch.bool, device=self.device),
        }
        return contacts, contact_log

    def process_joints(self):
        joint_log = {"count": 0, "indices": [], "error": [], "Js": []}
        c_body_idx, c_neigh, c_jac, c_error = [], [], [], []
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
                c_error.append(error)
                if joint.parent_idx != -1:
                    c_body_idx.append(joint.parent_idx)
                    c_neigh.append(joint.child_idx)
                    c_jac.append(J_2)
                    c_error.append(-error)
        joints = {
            "body_idx": torch.tensor(c_body_idx, dtype=torch.long, device=self.device),
            "neighbor_idx": torch.tensor(c_neigh, dtype=torch.long, device=self.device),
            "error": torch.tensor(c_error, dtype=torch.float32, device=self.device),
            "jac": (
                torch.stack(c_jac).to(self.device)
                if len(c_jac) > 0
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
            "dist": torch.cat([contacts["dist"], joints["error"]]),
            "is_equality": torch.cat(
                [
                    contacts["is_equality"],
                    torch.ones_like(joints["body_idx"], dtype=torch.bool, device=self.device),
                ]
            ),
            "counts": counts,
        }
        return merged

    def update_gnn_data(self, state: torch.Tensor, contacts: torch.Tensor):
        gnn_data = HeteroData()
        gnn_data["object"].x = torch.zeros(
            (self.num_shapes, 6), dtype=torch.float32, device=self.device
        )
        for i in range(self.num_shapes):
            gnn_data["object"].x[i][:] = torch.tensor(
                [
                    self.shapes[i].restitution,
                    self.shapes[i].mass,
                    state[i][0],
                    state[i][1],
                    torch.norm(state[i][:2]),
                    state[i][2],
                ],
                device=self.device,
            )

        if self.floor is not None:
            gnn_data["floor"].x = torch.tensor(
                [[self.floor.restitution]], dtype=torch.float32, device=self.device
            )
        else:
            gnn_data["floor"].x = torch.zeros((0, 1), dtype=torch.float32, device=self.device)

        all_edge_attrs = torch.cat([contacts["jac"], contacts["dist"].unsqueeze(1)], dim=1)
        mask_floor = contacts["neighbor_idx"] == -1
        mask_obj = ~mask_floor

        if mask_floor.any():
            target_nodes = contacts["body_idx"][mask_floor]
            source_nodes = torch.zeros_like(target_nodes)

            gnn_data["floor", "contact", "object"].edge_index = torch.stack(
                [source_nodes, target_nodes], dim=0
            )
            gnn_data["floor", "contact", "object"].edge_attr = all_edge_attrs[mask_floor]
        else:
            gnn_data["floor", "contact", "object"].edge_attr = torch.zeros(
                (0, 4), dtype=torch.float32, device=self.device
            )
            gnn_data["floor", "contact", "object"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            )

        if mask_obj.any():
            target_nodes = contacts["body_idx"][mask_obj]
            source_nodes = contacts["neighbor_idx"][mask_obj]

            gnn_data["object", "contact", "object"].edge_index = torch.stack(
                [source_nodes, target_nodes], dim=0
            )
            gnn_data["object", "contact", "object"].edge_attr = all_edge_attrs[mask_obj]
        else:
            gnn_data["object", "contact", "object"].edge_attr = torch.zeros(
                (0, 4), dtype=torch.float32, device=self.device
            )
            gnn_data["object", "contact", "object"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            )

        return gnn_data

    def state_from_gnn(self, gnn_output: tuple, contacts: list) -> torch.Tensor:
        object_states, lambdas_dict = gnn_output
        state_guess = torch.zeros(self.solver.state_shape(contacts), device=self.device)
        state_guess[:, :3] = object_states
        lambdas_obj = lambdas_dict[("object", "contact", "object")].view(-1)
        lambdas_floor = lambdas_dict[("floor", "contact", "object")].view(-1)
        if contacts["body_idx"].numel() > 0:
            mask_floor = contacts["neighbor_idx"] == -1
            mask_obj = ~mask_floor
            if mask_floor.any():
                body_idxs = contacts["body_idx"][mask_floor]
                local_idxs = contacts["local_idx"][mask_floor]
                n_floor = body_idxs.shape[0]
                state_guess[body_idxs, 3 + local_idxs] = lambdas_floor[:n_floor]
            if mask_obj.any():
                body_idxs = contacts["body_idx"][mask_obj]
                local_idxs = contacts["local_idx"][mask_obj]
                n_obj = body_idxs.shape[0]
                state_guess[body_idxs, 3 + local_idxs] = lambdas_obj[:n_obj]

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
            gnn_data = self.update_gnn_data(state, constraints)
            gnn_output = self.gnn(
                gnn_data.x_dict, gnn_data.edge_index_dict, gnn_data.edge_attr_dict
            )
            state_guess = self.state_from_gnn(gnn_output, constraints)

        return state_guess

    @abstractmethod
    def build_model(self):
        """
        Fill the self.shapes list with instances of classes from src.shapes and set self.floor
        """
        pass
