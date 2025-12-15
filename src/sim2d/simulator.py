import torch
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
        self.logger.log_init_config(self.shapes, self.floor, self.gravity, self.dt)

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
                    state = self.solver.step(state, contacts)
                with self.logger.timed_block("update_shapes"):
                    self.update_shapes(state)
        contacts, contact_log = self.collide()
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
                    contacts[i].append((distance, J_1))
                    if self.logger.config.enable_hdf5:
                        contact_log["count"] += 1
                        contact_log["distances"].append(distance)
                        contact_log["Js"].append((J_1, J_2))
                    if not isinstance(shape_2, Floor):
                        contacts[j].append((distance, J_2))
                        if self.logger.config.enable_hdf5:
                            contact_log["indices"].append((i, j))
                    elif self.logger.config.enable_hdf5:
                        contact_log["indices"].append((i, -1))

        return contacts, contact_log

    @abstractmethod
    def build_model(self):
        """
        Fill the self.shapes list with instances of classes from src.shapes and set self.floor
        """
        pass

    @abstractmethod
    def init_state_fn(self, state: torch.Tensor, contacts: torch.Tensor, dt: float):
        """
        return guess for next state of shape (self.num_shapes x 3+max([len(a) for a in contacts]))
        """
        state_guess = torch.zeros(self.solver.state_shape(contacts))
        state_guess[:, :3] += state[:, :3]
        for i in range(self.num_shapes):
            if not isinstance(self.shapes[i], Floor):
                state_guess[i, :3] += dt * self.gravity
        return state_guess
