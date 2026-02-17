import argparse
import os
from pathlib import Path

import numpy as np
import torch
from h5py import File

import sim2d
from sim2d.utils import SceneGenerator
from sim2d import Circle, RevoluteJoint, FixedJoint


class CustomDatasetGenerator(SceneGenerator):
    def __init__(self, steps, gravity, dt, logging_config):
        self.max_attempts = 2000
        sim_time = steps * dt
        newton_iters = 500
        device = "cpu"
        super().__init__(
            self.max_attempts,
            sim_time,
            newton_iters,
            gravity,
            dt,
            logging_config=logging_config,
            device=device,
        )
        self.num_steps = steps  # make sure there is no division error when calculating sim_time

    def build_model(self):
        self.add_floor()

        num_shapes = np.random.randint(1, 10)
        shapes_placed = 0
        attempts = 0

        while (shapes_placed < num_shapes) and (attempts < self.max_attempts):
            attempts += 1
            strategy_probs = [0.2, 0.2, 0.3, 0.3]
            strategy = np.random.choice([0, 1, 2, 3], p=strategy_probs)
            if shapes_placed == 0 and strategy in [2, 3]:
                strategy = np.random.choice([0, 1])
            succ = False
            if strategy == 0:
                succ = self.add_shape(Circle)
            elif strategy == 1:
                succ = self.add_shape_floor_contact(Circle)
            elif strategy == 2:
                succ = self.add_shape_shape_contact(None, Circle)
            elif strategy == 3:
                succ = (
                    self.add_shape(Circle)
                    if np.random.random() < 0.5
                    else self.add_shape_floor_contact(Circle)
                )
                if succ:
                    child_idx = len(self.shapes) - 1
                    valid_parents = [-1] + list(range(len(self.shapes) - 1))
                    parent_idx = np.random.choice(valid_parents)
                    succ = self.add_joint(
                        RevoluteJoint if np.random.random() < 0.5 else FixedJoint,
                        child_idx,
                        parent_idx,
                        0.5,
                    )
                    if not succ:
                        self.shapes.pop()
            if succ:
                shapes_placed += 1


def sim_stable(filepath: str, threshold: float = 1e2) -> bool:
    def nan_or_inf(arr):
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return True
        return False

    with File(filepath, "r") as f:
        for step in (k for k in f.keys() if k.startswith("step_")):
            if "contacts_data" in f[step]:
                lambdas = f[step]["contacts_data"]["lambdas"][:]
                if lambdas.size > 0:
                    if np.max(np.abs(lambdas)) > threshold:
                        return False
            if "joint_data" in f[step]:
                lambdas_j = f[step]["joint_data"]["lambdas"][:]
                if lambdas_j.size > 0:
                    if np.max(np.abs(lambdas_j)) > threshold:
                        return False
            if (
                nan_or_inf(f[step]["shapes_data"]["translation"][:])
                or nan_or_inf(f[step]["shapes_data"]["velocity"][:])
                or nan_or_inf(f[step]["shapes_data"]["rotation"][:])
                or nan_or_inf(f[step]["shapes_data"]["angular_velocity"][:])
            ):
                return False
    return True


def create_dataset(start_pass_idx: int, num_passes: int, dataset_path: Path):
    steps = np.random.randint(50, 100)
    dt = 1 / 100
    gravity = torch.tensor([0.0, -9.81, 0.0])
    i = start_pass_idx
    while i < start_pass_idx + num_passes:
        hdf5_path = dataset_path / "raw" / f"pass_{i}.h5"
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        logging_config = sim2d.LoggingConfig(False, True, False, hdf5_path)
        sim = CustomDatasetGenerator(steps, gravity, dt, logging_config)
        try:
            sim.run()
            if sim_stable(hdf5_path):
                print(f"Generated pass {i}")
                i += 1
            else:
                print(f"Pass {i} unstable, retrying...")
                os.remove(hdf5_path)
        except Exception as e:
            print(f"Simulation failed: {e}")
            if hdf5_path.exists():
                os.remove(hdf5_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Training Data with Custom Scene Generator"
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_passes", type=int, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    args = parser.parse_args()

    create_dataset(args.start_idx, args.num_passes, args.dataset_path)
