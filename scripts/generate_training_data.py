from pathlib import Path
import sys

import torch
import numpy as np

import sim2d
from sim2d.gnn import DatasetSim2D

np.random.seed(0)


class SimulatorGenerator(sim2d.Simulator):
    def __init__(self, logging_config):
        newton_iters = 50
        gravity = torch.tensor([0.0, -9.81, 0.0])
        dt = 1 / np.random.randint(50, 500)
        sim_time = np.random.randint(50, 500) * dt
        super().__init__(sim_time, newton_iters, gravity, dt, logging_config)

    def build_model(self):
        self.floor = sim2d.Floor(np.random.uniform(-1.0, 1.0), np.random.random())
        num_shapes = np.random.randint(1, 13)
        max_attempts = 1000
        shapes_placed = 0
        max_collision = 0.05
        attempts = 0
        while (shapes_placed < num_shapes) and (attempts < max_attempts):
            translation = torch.tensor(
                np.random.uniform(self.floor.height - 0.1, self.floor.height + 2.5, 2)
            )
            rotation = torch.tensor(np.random.uniform(0, 2 * np.pi))
            velocity = torch.tensor(np.random.uniform(-1.0, 1.0, 2))
            angular_velocity = torch.tensor(np.random.uniform(-np.pi / 2, np.pi / 2))
            mass = np.random.uniform(0.1, 5.0)
            restitution = np.random.random()
            if np.random.random() <= 0.15:
                shape = sim2d.Point(translation, velocity, mass, restitution)
            else:
                radius = np.random.uniform(0.05, 0.5)
                shape = sim2d.Circle(translation, velocity, mass, restitution, radius)
            contacts = [sim2d.compute_collision(shape, s)[1] for s in self.shapes]
            contacts.append(sim2d.compute_collision(shape, self.floor)[1])
            attempts += 1
            if (len(contacts) == 0) or (max(contacts) < max_collision):
                self.shapes.append(shape)
                shapes_placed += 1

    def init_state_fn(self, state, contacts, dt):
        return super().init_state_fn(state, contacts, dt)


def create_dataset(start_pass_idx: int, num_passes: int, dataset_path: Path):
    for i in range(start_pass_idx, start_pass_idx + num_passes):
        logging_config = sim2d.LoggingConfig(False, True, dataset_path / f"pass_{i}.h5")
        sim = SimulatorGenerator(logging_config)
        sim.run()


if __name__ == "__main__":
    start_pass_idx = int(sys.argv[1])
    num_passes = int(sys.argv[2])
    dataset_path = Path(sys.argv[3])
    create_dataset(start_pass_idx, num_passes, dataset_path)
    DatasetSim2D(dataset_path, True)
