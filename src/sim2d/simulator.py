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
        self.num_steps = int(sim_time // dt + sim_time % dt)
        self.newton_iters = newton_iters
        self.gravity = gravity
        self.dt = dt
        self.shapes: list[Shape] = []
        self.floor = None

        if logging_config is None:
            logging_config = LoggingConfig()
        self.logger = EngineLogger(logging_config)

        self.build_model()
        self.num_shapes = len(self.shapes)
        assert self.num_shapes > 0, "Cannot simulate nothing"
        assert not Floor in [type(s) for s in self.shapes], "Floor should be saved in self.floor"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn = None
        if not init_gnn_path is None:
            self.gnn = torch.load(init_gnn_path, self.device, weights_only=False)
            self.gnn_data = self.prepare_gnn_data()
            self.gnn_data = self.gnn_data.to(self.device)

        self.solver = EulerSolver(
            self.shapes,
            self.newton_iters,
            self.gravity,
            self.dt,
            self.init_state_fn,
            self.logger,
        )

    def run(self):
        self.logger.open()
        self.logger.log_init_config(self)

        state = torch.zeros((self.num_shapes, 3))
        for i in range(self.num_shapes):
            state[i, :] = torch.cat(
                [self.shapes[i].velocity, torch.tensor([self.shapes[i].angular_velocity])]
            )
        with torch.no_grad():
            for i in tqdm(range(self.num_steps), desc="Simulation"):
                current_time = i * self.dt
                with self.logger.timed_block("collision_detection"):
                    contacts, contact_log = self.collide()
                self.logger.log_step_data(i, current_time, self.shapes, state, contact_log)
                with self.logger.timed_block("physics_step"):
                    state = self.solver.step(i, state, contacts)
                with self.logger.timed_block("update_shapes"):
                    self.update_shapes(state)
        _, contact_log = self.collide()
        self.logger.log_step_data(i + 1, current_time + self.dt, self.shapes, state, contact_log)
        self.logger.close()

    def update_shapes(self, state):
        for i in range(self.num_shapes):
            self.shapes[i].translation += state[i][:2] * self.dt
            self.shapes[i].rotation += state[i][2] * self.dt
            self.shapes[i].velocity = state[i][:2]
            self.shapes[i].angular_velocity = state[i][2]

    def collide(self) -> list[list[tuple[float, torch.Tensor]]]:
        contact_log = {"count": 0, "indices": [], "distances": [], "Js": []}
        contacts = [[] for _ in range(self.num_shapes)]
        shapes = self.shapes + [self.floor] if not self.floor is None else self.shapes
        if len(shapes) < 2:
            return contacts, contact_log
        for i, shape_1 in enumerate(shapes):
            for j, shape_2 in enumerate(shapes[i + 1 :], i + 1):
                in_collision, distance, J_1, J_2 = compute_collision(shape_1, shape_2)
                if in_collision:
                    if self.logger.config.enable_hdf5:
                        contact_log["count"] += 1
                        contact_log["distances"].append(distance)
                        contact_log["Js"].append((J_1, J_2))
                    if not isinstance(shape_2, Floor):
                        contacts[i].append((distance, J_1, j))
                        contacts[j].append((distance, J_2, i))
                        if self.logger.config.enable_hdf5:
                            contact_log["indices"].append((i, j))
                    elif isinstance(shape_2, Floor):
                        contacts[i].append((distance, J_1, -1))
                        if self.logger.config.enable_hdf5:
                            contact_log["indices"].append((i, -1))

        return contacts, contact_log

    def prepare_gnn_data(self):
        gnn_data = HeteroData()
        gnn_data["world"].x = torch.tensor(
            [[self.dt, self.gravity[0], self.gravity[1], self.gravity[2]]], dtype=torch.float32
        )

        nodes_object = []
        attrs_world_object = []
        indices_world_object = []
        for i in range(self.num_shapes):
            nodes_object.append(
                [self.shapes[i].restitution, self.shapes[i].mass, 0.0, 0.0, 0.0, 0.0]
            )
            attrs_world_object.append([0.0, 0.0, 0.0, 0.0])
            indices_world_object.append([0, i])
        gnn_data["object"].x = torch.tensor(nodes_object, dtype=torch.float32)
        gnn_data["world", "w2o", "object"].edge_attr = torch.tensor(
            attrs_world_object, dtype=torch.float32
        )
        gnn_data["world", "w2o", "object"].edge_index = torch.tensor(
            indices_world_object, dtype=torch.long
        ).T

        if self.floor is not None:
            gnn_data["floor"].x = torch.tensor([[self.floor.restitution]], dtype=torch.float32)
            gnn_data["world", "w2f", "floor"].edge_attr = torch.tensor(
                [[self.floor.height]], dtype=torch.float32
            )
            gnn_data["world", "w2f", "floor"].edge_index = torch.tensor(
                [[0, 0]], dtype=torch.long
            ).T
        else:
            gnn_data["floor"].x = torch.zeros((0, 1), dtype=torch.float32)
            gnn_data["world", "w2f", "floor"].edge_attr = torch.zeros((0, 1), dtype=torch.float32)
            gnn_data["world", "w2f", "floor"].edge_index = torch.zeros((2, 0), dtype=torch.long)

        return gnn_data

    def update_gnn_data(self, state: torch.Tensor, contacts: torch.Tensor):
        for i in range(self.num_shapes):
            self.gnn_data["object"].x[i][2:] = torch.tensor(
                [state[i][0], state[i][1], torch.norm(state[i][:2]), state[i][2]],
                device=self.device,
            )

            self.gnn_data["world", "w2o", "object"].edge_attr[i][:] = torch.tensor(
                [
                    self.shapes[i].translation[0],
                    self.shapes[i].translation[1],
                    torch.norm(self.shapes[i].translation),
                    self.shapes[i].rotation,
                ],
                device=self.device,
            )
        attrs_object_object = []
        indices_object_object = []
        attrs_floor_object = []
        indices_floor_object = []
        for i in range(self.num_shapes):
            for contact in contacts[i]:
                dist, J, neighbor_idx = contact
                edge_attr = [J[0], J[1], J[2], dist]
                if neighbor_idx == -1:
                    attrs_floor_object.append(edge_attr)
                    indices_floor_object.append([0, i])
                else:
                    attrs_object_object.append(edge_attr)
                    indices_object_object.append([neighbor_idx, i])
        if len(indices_object_object) > 0:
            self.gnn_data["object", "contact", "object"].edge_attr = torch.tensor(
                attrs_object_object, dtype=torch.float32, device=self.device
            )
            self.gnn_data["object", "contact", "object"].edge_index = torch.tensor(
                indices_object_object, dtype=torch.long, device=self.device
            ).T
        else:
            self.gnn_data["object", "contact", "object"].edge_attr = torch.zeros(
                (0, 4), dtype=torch.float32, device=self.device
            )
            self.gnn_data["object", "contact", "object"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            )

        if len(indices_floor_object) > 0:
            self.gnn_data["floor", "contact", "object"].edge_attr = torch.tensor(
                attrs_floor_object, dtype=torch.float32, device=self.device
            )
            self.gnn_data["floor", "contact", "object"].edge_index = torch.tensor(
                indices_floor_object, dtype=torch.long, device=self.device
            ).T
        else:
            self.gnn_data["floor", "contact", "object"].edge_attr = torch.zeros(
                (0, 4), dtype=torch.float32, device=self.device
            )
            self.gnn_data["floor", "contact", "object"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long, device=self.device
            )

    def state_from_gnn(self, gnn_output: tuple, contacts: list) -> torch.Tensor:
        object_states, lambdas_dict = gnn_output
        state_guess = torch.zeros(self.solver.state_shape(contacts))
        state_guess[:, :3] = object_states
        lambdas_obj = lambdas_dict[("object", "contact", "object")].view(-1)
        lambdas_floor = lambdas_dict[("floor", "contact", "object")].view(-1)
        idx_obj = 0
        idx_floor = 0
        for i in range(self.num_shapes):
            for j, contact in enumerate(contacts[i]):
                neighbor_idx = contact[2]
                if neighbor_idx == -1:
                    state_guess[i, 3 + j] = lambdas_floor[idx_floor]
                    idx_floor += 1
                else:
                    state_guess[i, 3 + j] = lambdas_obj[idx_obj]
                    idx_obj += 1
        return state_guess

    def init_state_fn(self, state: torch.Tensor, contacts: torch.Tensor, dt: float):
        """
        return guess for next state of shape (self.num_shapes x 3+max([len(a) for a in contacts]))
        """
        if self.gnn is None:
            state_guess = torch.zeros(self.solver.state_shape(contacts))
            state_guess[:, :3] += state[:, :3]
            for i in range(self.num_shapes):
                if not isinstance(self.shapes[i], Floor):
                    state_guess[i, :3] += dt * self.gravity
        else:
            self.update_gnn_data(state, contacts)
            gnn_output = self.gnn(
                self.gnn_data.x_dict, self.gnn_data.edge_index_dict, self.gnn_data.edge_attr_dict
            )
            state_guess = self.state_from_gnn(gnn_output, contacts)

        return state_guess

    @abstractmethod
    def build_model(self):
        """
        Fill the self.shapes list with instances of classes from src.shapes and set self.floor
        """
        pass
