from typing import Optional

from pathlib import Path
import numpy as np
import torch

import sim2d
from sim2d.logger import LoggingConfig
from sim2d.shapes import Point, Circle, Rectangle, Floor
from sim2d.joints import FixedJoint, RevoluteJoint, PrismaticJoint
from sim2d.collisions import compute_collision, _get_support_point


class SceneGenerator(sim2d.Simulator):
    def __init__(
        self,
        max_attempts,
        sim_time,
        newton_iters: int = 100,
        gravity: torch.Tensor = torch.tensor([0.0, -9.81, 0.0]),
        dt: float = 0.01,
        init_gnn_path: Optional[str | Path] = None,
        logging_config: Optional[LoggingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.max_attempts = max_attempts
        self.device = torch.device(device)
        super().__init__(sim_time, newton_iters, gravity, dt, init_gnn_path, logging_config, device)

    def _get_random_shape_params(self):
        translation = torch.tensor(np.random.uniform(-2.0, 2.0, 2), dtype=torch.float32)
        velocity = torch.tensor(np.random.uniform(-1.0, 1.0, 2), dtype=torch.float32)
        mass = torch.tensor(np.random.uniform(0.5, 5.0), dtype=torch.float32)
        restitution = torch.tensor(np.random.uniform(0.1, 0.9), dtype=torch.float32)
        rotation = torch.tensor(np.random.uniform(0, 2 * np.pi), dtype=torch.float32)
        angular_velocity = torch.tensor(np.random.uniform(-np.pi, np.pi), dtype=torch.float32)
        return translation, velocity, mass, restitution, rotation, angular_velocity

    def _create_shape_instance(self, shape_type):
        t, v, m, rest, rot, w = self._get_random_shape_params()
        if shape_type is None:
            shape_type = np.random.choice([Point, Circle, Rectangle])

        if shape_type == Point:
            return Point(t, rot, v, w, m, rest)
        elif shape_type == Circle:
            radius = torch.tensor(np.random.uniform(0.1, 0.5), dtype=torch.float32)
            return Circle(t, rot, v, w, m, rest, radius)
        elif shape_type == Rectangle:
            sides = torch.tensor(np.random.uniform(0.2, 1.0, 2), dtype=torch.float32)
            return Rectangle(t, rot, v, w, m, rest, sides)
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")

    def _is_spawn_valid(
        self, new_shape, expected_contact_idx=None, expected_floor_contact=False, max_depth=0.05
    ):
        if self.floor is not None:
            active, dist, _, _ = compute_collision(new_shape, self.floor, self.device)
            if active:
                if not expected_floor_contact:
                    return False
                elif dist.item() > max_depth:
                    return False
            elif expected_floor_contact:
                return False

        contact_achieved = False if expected_contact_idx is not None else True

        for i, existing_shape in enumerate(self.shapes):
            active, dist, _, _ = compute_collision(new_shape, existing_shape, self.device)
            if active:
                if expected_contact_idx == i:
                    if dist.item() > max_depth:
                        return False
                    contact_achieved = True
                else:
                    return False

        return contact_achieved

    def _get_shape_support_local(self, shape, direction: torch.Tensor):
        """Calculates the support point of a shape in local coordinates (relative to its center)."""
        if isinstance(shape, (Point, Circle)):
            radius = getattr(shape, "radius", torch.tensor(0.0, device=self.device))
            return radius * direction
        elif isinstance(shape, Rectangle):
            orig_trans = shape.translation.clone()
            shape.translation = torch.zeros(2, dtype=torch.float32, device=self.device)
            sup = _get_support_point(shape, direction)
            shape.translation = orig_trans
            return sup
        else:
            raise ValueError("Unknown shape")

    def add_floor(self):
        self.floor = Floor(
            height=torch.tensor(np.random.uniform(1.0, 2.0)),
            restitution=torch.tensor(np.random.uniform(0.0, 1.0)),
        )

    def add_shape(self, shape_type=None) -> bool:
        for _ in range(self.max_attempts):
            new_shape = self._create_shape_instance(shape_type)
            if self._is_spawn_valid(new_shape):
                self.shapes.append(new_shape)
                return True
        return False

    def add_shape_shape_contact(self, target_idx=None, shape_type=None, max_depth=0.07) -> bool:
        if not self.shapes:
            raise ValueError("No shapes exist to form a contact with.")

        if target_idx is None:
            target_idx = np.random.randint(len(self.shapes))

        target_shape = self.shapes[target_idx]

        for _ in range(self.max_attempts):
            new_shape = self._create_shape_instance(shape_type)

            theta = np.random.uniform(0, 2 * np.pi)
            direction = torch.tensor(
                [np.cos(theta), np.sin(theta)], dtype=torch.float32, device=self.device
            )

            if isinstance(target_shape, (Point, Circle)):
                r1 = getattr(target_shape, "radius", torch.tensor(0.0, device=self.device))
                p1_world = target_shape.translation + r1 * direction
            else:
                p1_world = _get_support_point(target_shape, direction)

            p2_local = self._get_shape_support_local(new_shape, -direction)

            depth = np.random.uniform(1e-4, max_depth)
            new_shape.translation = p1_world - (depth * direction) - p2_local

            if self._is_spawn_valid(
                new_shape, expected_contact_idx=target_idx, max_depth=max_depth
            ):
                self.shapes.append(new_shape)
                return True

        return False

    def add_shape_floor_contact(self, shape_type=None, max_depth=0.07) -> bool:
        if self.floor is None:
            raise ValueError("Floor is not initialized.")

        for _ in range(self.max_attempts):
            new_shape = self._create_shape_instance(shape_type)
            direction = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.device)
            x_pos = np.random.uniform(-2.0, 2.0)
            p1_world = torch.tensor(
                [x_pos, self.floor.height.item()],
                dtype=torch.float32,
                device=self.device,
            )

            p2_local = self._get_shape_support_local(new_shape, -direction)
            depth = np.random.uniform(1e-4, max_depth)
            new_shape.translation = p1_world - (depth * direction) - p2_local

            if self._is_spawn_valid(new_shape, expected_floor_contact=True, max_depth=max_depth):
                self.shapes.append(new_shape)
                return True

        return False

    def _rotate_vec(self, vec: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(angle), torch.sin(angle)
        x = c * vec[0] - s * vec[1]
        y = s * vec[0] + c * vec[1]
        return torch.stack([x, y])

    def _is_point_inside_shape(self, local_point: torch.Tensor, shape) -> bool:
        """Checks if a given local coordinate point is inside the shape's boundaries."""
        if isinstance(shape, (Point, Circle)):
            radius = getattr(shape, "radius", torch.tensor(0.0, device=self.device))
            return torch.norm(local_point) < radius
        elif isinstance(shape, Rectangle):
            half_w = shape.sides[0] / 2.0
            half_h = shape.sides[1] / 2.0
            return (torch.abs(local_point[0]) < half_w) and (torch.abs(local_point[1]) < half_h)
        elif isinstance(shape, Floor):
            return local_point[1] < 0.0
        return False

    def add_joint(
        self, joint_type=None, shape_idx_1=None, shape_idx_2=None, coincide_prob=0.5
    ) -> bool:
        if not self.shapes:
            raise ValueError("Not enough shapes to create a joint.")

        if shape_idx_1 is None:
            shape_idx_1 = np.random.randint(len(self.shapes))

        if shape_idx_2 is None:
            valid_targets = [-1] + [i for i in range(len(self.shapes)) if i != shape_idx_1]
            shape_idx_2 = np.random.choice(valid_targets)

        if joint_type is None:
            joint_type = np.random.choice([FixedJoint, RevoluteJoint, PrismaticJoint])

        shape_1 = self.shapes[shape_idx_1]
        shape_2 = self.shapes[shape_idx_2] if shape_idx_2 != -1 else self.floor

        anchors_valid = False

        for _ in range(self.max_attempts):
            world_anchor = (shape_1.translation + shape_2.translation) / 2.0
            world_anchor += torch.tensor(
                np.random.uniform(-1.0, 1.0, 2),
                dtype=torch.float32,
                device=self.device,
            )

            if shape_idx_2 == -1 and self.floor is not None:
                world_anchor[1] = torch.max(world_anchor[1], self.floor.height)

            child_anchor = self._rotate_vec(world_anchor - shape_1.translation, -shape_1.rotation)
            if np.random.random() > coincide_prob:
                gaussian_std_dev = 0.1
                world_anchor = (
                    world_anchor
                    + torch.randn(2, dtype=torch.float32, device=self.device) * gaussian_std_dev
                )

            if shape_idx_2 == -1:
                parent_anchor = world_anchor
            else:
                parent_anchor = self._rotate_vec(
                    world_anchor - shape_2.translation, -shape_2.rotation
                )

            child_inside = self._is_point_inside_shape(child_anchor, shape_1)
            parent_inside = False
            if shape_2 is not None:
                parent_inside = self._is_point_inside_shape(parent_anchor, shape_2)

            if not child_inside and not parent_inside:
                anchors_valid = True
                break

        if not anchors_valid:
            return False

        if joint_type == FixedJoint:
            c_tr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            p_tr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            joint = FixedJoint(shape_idx_1, shape_idx_2, child_anchor, parent_anchor, c_tr, p_tr)
        elif joint_type == PrismaticJoint:
            axis = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device)
            axis = self._rotate_vec(
                axis, torch.tensor(np.random.uniform(0, 2 * np.pi), device=self.device)
            )
            joint = PrismaticJoint(shape_idx_1, shape_idx_2, child_anchor, parent_anchor, axis)
        elif joint_type == RevoluteJoint:
            joint = RevoluteJoint(shape_idx_1, shape_idx_2, child_anchor, parent_anchor)
        else:
            raise ValueError(f"Unsupported joint type: {joint_type}")

        self.joints.append(joint)
        return True
