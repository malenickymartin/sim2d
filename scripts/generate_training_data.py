import argparse
import os

from pathlib import Path
import torch
import numpy as np
from h5py import File

import sim2d
from sim2d import Shape
from sim2d.engine import EngineLogger

np.random.seed(0)


def rotate_vec(vec: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    angle = torch.as_tensor(angle)
    c, s = torch.cos(angle), torch.sin(angle)
    x = c * vec[0] - s * vec[1]
    y = s * vec[0] + c * vec[1]
    return torch.stack([x, y])


class SimulatorGenerator(sim2d.Simulator):
    def __init__(self):
        newton_iters = 500
        gravity = torch.tensor([0.0, -9.81, 0.0])
        dt = 1 / 100
        warmup_steps = 20
        sim_time = warmup_steps * dt
        self.device = torch.device("cpu")
        logging_config = sim2d.LoggingConfig(False, False, False, None)
        super().__init__(
            sim_time, newton_iters, gravity, dt, logging_config=logging_config, device=self.device
        )

    def build_model(self):
        self.floor = sim2d.Floor(np.random.uniform(-1.0, 1.0), np.random.random())
        num_shapes = np.random.randint(1, 13)
        max_attempts = 1000
        shapes_placed = 0
        attempts = 0
        while (shapes_placed < num_shapes) and (attempts < max_attempts):
            attempts += 1
            shape = self.generate_shape_properties()
            parent_idx = None
            if np.random.random() < 0.3:
                parent_idx = np.random.choice([-1] + list(range(len(self.shapes))))
                parent_shape = self.shapes[parent_idx] if parent_idx != -1 else self.floor
                new_joint = self.setup_joint_connection(shape, parent_shape, parent_idx)
            else:
                shape.translation = torch.tensor(
                    [
                        np.random.uniform(-1.0, 1.0),
                        np.random.uniform(
                            self.floor.height + shape.radius - 0.05, self.floor.height + 2.0
                        ),
                    ]
                )
                shape.rotation = torch.tensor(np.random.uniform(0, 2 * np.pi))

            succ = self.check_spawn_validity(shape, parent_idx)
            if succ:
                self.shapes.append(shape)
                if parent_idx is not None:
                    new_joint.child_idx = len(self.shapes) - 1
                    self.joints.append(new_joint)
                shapes_placed += 1

    def generate_shape_properties(self):
        velocity = torch.tensor(np.random.uniform(-1.0, 1.0, 2))
        angular_velocity = torch.tensor(np.random.uniform(-np.pi / 2, np.pi / 2))
        mass = np.random.uniform(0.1, 5.0)
        restitution = np.random.random()
        translation = torch.zeros(2)
        rotation = torch.tensor(0.0)
        if np.random.random() <= 0.15:
            return sim2d.Point(translation, velocity, mass, restitution)
        else:
            radius = np.random.uniform(0.05, 0.5)
            return sim2d.Circle(translation, velocity, mass, restitution, radius)

    def setup_joint_connection(self, child_shape: Shape, parent_shape: Shape, parent_idx: int):
        def _sample_stadium(p1, p2, radius):
            vec = p2 - p1
            L = torch.norm(vec)
            unit_vec = vec / L
            normal_vec = rotate_vec(unit_vec, torch.pi / 2)
            area_rect = 2 * radius * L
            area_circ = np.pi * radius**2
            total_area = area_rect + area_circ
            prob_rect = area_rect / total_area
            if np.random.random() < prob_rect:
                u_l = np.random.uniform(0, L)
                u_w = np.random.uniform(-radius, radius)
                local_point = (u_l * unit_vec) + (u_w * normal_vec)
            else:
                r_rand = radius * torch.sqrt(torch.tensor(np.random.random()))
                theta_rand = torch.tensor(np.random.uniform(0, 2 * np.pi))
                x_c = r_rand * torch.cos(theta_rand)
                y_c = r_rand * torch.sin(theta_rand)
                if x_c >= 0:
                    x_c += L
                local_point = (x_c * unit_vec) + (y_c * normal_vec)
            return p1 + local_point

        child_shape.rotation = torch.tensor(np.random.uniform(0, 2 * np.pi))
        parent_separation = parent_shape.radius if parent_idx != -1 else 0.0
        max_attempts = 1000
        attempts = 0
        succ = False
        while not succ and (attempts < max_attempts):
            child_shape.translation = parent_shape.translation + (
                parent_separation + child_shape.radius + np.random.uniform(0, 1.0)
            ) * rotate_vec(torch.tensor([1.0, 0.0]), np.random.uniform(0.0, 2 * np.pi))
            succ = self.check_spawn_validity(child_shape, parent_idx)
        if parent_idx != -1:
            anchor = _sample_stadium(parent_shape.translation, child_shape.translation, 0.5)
            parent_anchor = rotate_vec(anchor - parent_shape.translation, -parent_shape.rotation)
        else:
            anchor = torch.tensor(np.random.uniform(0.0, 2.0, 2)) + parent_shape.translation
            parent_anchor = anchor.clone()
        child_anchor = rotate_vec(anchor - child_shape.translation, -child_shape.rotation)

        JointClass = np.random.choice([sim2d.RevoluteJoint, sim2d.FixedJoint, sim2d.PrismaticJoint])
        if JointClass == sim2d.FixedJoint:
            child_shape.velocity = parent_shape.velocity.clone()
            child_shape.angular_velocity = parent_shape.angular_velocity.clone()
            r = np.random.random()
            rot_target = parent_shape.rotation - child_shape.rotation
            c_tr = r * rot_target
            p_tr = -(1 - r) * rot_target
            joint = sim2d.FixedJoint(0, parent_idx, child_anchor, parent_anchor, c_tr, p_tr)
        else:
            joint = sim2d.RevoluteJoint(0, parent_idx, child_anchor, parent_anchor)

        return joint

    def check_spawn_validity(self, shape, parent_idx):
        """Checks if the new shape collides with the floor or existing shapes (excluding ignore_shape)."""
        max_collision_depth = 0.05
        _, dist_floor, _, _ = sim2d.compute_collision(shape, self.floor, self.device)
        if dist_floor > max_collision_depth or (parent_idx == -1 and dist_floor > 0.0):
            return False
        for i, s in enumerate(self.shapes):
            _, dist, _, _ = sim2d.compute_collision(shape, s, self.device)
            if dist > max_collision_depth or (parent_idx == i and dist > 0.0):
                return False
        return True


def sim_stable(filepath: str, threshold: float = 1e3):
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
    i = start_pass_idx
    while i < start_pass_idx + num_passes:
        hdf5_path = dataset_path / "raw" / f"pass_{i}.h5"
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        logging_config = sim2d.LoggingConfig(False, True, False, hdf5_path)
        engine_logger = EngineLogger(logging_config)
        sim = SimulatorGenerator()
        try:
            sim.run()
            sim.solver.logger = engine_logger
            sim.logger = engine_logger
            sim.num_steps = np.random.randint(20, 50)
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
    parser = argparse.ArgumentParser(description="Generate Training Data")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_passes", type=int, required=True)
    parser.add_argument("--dataset_path", type=Path, required=True)
    args = parser.parse_args()
    create_dataset(args.start_idx, args.num_passes, args.dataset_path)
