from abc import ABC

import torch

from .shapes import Shape


class Joint(ABC):
    def __init__(
        self,
        child_idx: int,
        parent_idx: int,
        child_anchor: torch.Tensor,
        parent_anchor: torch.Tensor,
    ):
        self.child_idx: int = child_idx
        self.parent_idx: int = parent_idx  # If -1, parent is static world
        self.child_anchor = torch.as_tensor(child_anchor, dtype=torch.float32)
        self.parent_anchor = torch.as_tensor(parent_anchor, dtype=torch.float32)

    def to(self, device: torch.device):
        self.child_anchor = self.child_anchor.to(device)
        self.parent_anchor = self.parent_anchor.to(device)


class FixedJoint(Joint):
    def __init__(
        self,
        child_idx: int,
        parent_idx: int,
        child_anchor: torch.Tensor,
        parent_anchor: torch.Tensor,
        child_target_rotation: float = 0.0,
        parent_target_rotation: float = 0.0,
    ):
        super().__init__(child_idx, parent_idx, child_anchor, parent_anchor)
        self.child_target_rotation = torch.as_tensor(child_target_rotation, dtype=torch.float32)
        self.parent_target_rotation = torch.as_tensor(parent_target_rotation, dtype=torch.float32)

    def to(self, device: torch.device):
        self.child_target_rotation = self.child_target_rotation.to(device)
        self.parent_target_rotation = self.parent_target_rotation.to(device)
        super().to(device)


class PrismaticJoint(Joint):
    def __init__(
        self,
        child_idx: int,
        parent_idx: int,
        child_anchor: torch.Tensor,
        parent_anchor: torch.Tensor,
        axis: torch.Tensor = torch.Tensor([1.0, 0.0]),
    ):
        super().__init__(child_idx, parent_idx, child_anchor, parent_anchor)
        self.axis = torch.as_tensor(axis, dtype=torch.float32)

    def to(self, device: torch.device):
        self.axis = self.axis.to(device)
        super().to(device)


class RevoluteJoint(Joint):
    def __init__(
        self,
        child_idx,
        parent_idx,
        child_anchor: torch.Tensor,
        parent_anchor: torch.Tensor,
    ):
        super().__init__(child_idx, parent_idx, child_anchor, parent_anchor)


JOINT_TO_INT = {FixedJoint: 0, RevoluteJoint: 1, PrismaticJoint: 2}
INT_TO_JOINT = {v: k for k, v in JOINT_TO_INT.items()}
JOINT_INT_TO_STR = {0: "fixed_joint", 1: "revolute_joint", 2: "prismatic_joint"}
JOINT_NUM_CONSTR = {0: 3, 1: 2, 2: 2}


def joint_to_int(joint):
    joint_type = type(joint)
    assert (
        joint_type in JOINT_TO_INT.keys()
    ), f"Unknown joint type. Joint type: {joint_type}, known types: {JOINT_TO_INT.keys()}"
    return JOINT_TO_INT[joint_type]


def int_to_joint(i: int):
    assert (
        i in INT_TO_JOINT.keys()
    ), f"Unknown joint type. Joint type: {i}, known types: {INT_TO_JOINT.keys()}"
    return INT_TO_JOINT[i]


def _rot_matrix(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([c, -s, s, c]).view(2, 2)


def _perp(v):
    return torch.stack([-v[1], v[0]])


def compute_joint_constraints(
    joint: Joint, shape_1: Shape, shape_2: Shape | None, device: torch.device
):
    constraints = []

    p_1 = shape_1.translation
    theta_1 = shape_1.rotation
    R_1 = _rot_matrix(theta_1)

    if shape_2 is not None:
        p_2 = shape_2.translation
        theta_2 = shape_2.rotation
        R_2 = _rot_matrix(theta_2)
    else:
        p_2 = torch.zeros(2, device=device)
        theta_2 = torch.tensor(0.0, device=device)
        R_2 = torch.eye(2, device=device)

    r_1, r_2 = R_1 @ joint.child_anchor, R_2 @ joint.parent_anchor
    diff = (p_2 + r_2) - (p_1 + r_1)

    if isinstance(joint, RevoluteJoint):
        n_x = torch.tensor([1.0, 0.0], device=device)
        J1_x, J2_x = torch.zeros(3, device=device), torch.zeros(3, device=device)
        J1_x[:2] = n_x
        J1_x[2] = _perp(r_1).dot(n_x)
        J1_x = J1_x / J1_x.norm()
        if shape_2:
            J2_x[:2] = -n_x
            J2_x[2] = _perp(r_2).dot(-n_x)
            J2_x = J2_x / J2_x.norm()
        constraints.append((J1_x, J2_x, diff[0]))

        n_y = torch.tensor([0.0, 1.0], device=device)
        J1_y, J2_y = torch.zeros(3, device=device), torch.zeros(3, device=device)
        J1_y[:2] = n_y
        J1_y[2] = _perp(r_1).dot(n_y)
        J1_y = J1_y / J1_y.norm()
        if shape_2:
            J2_y[:2] = -n_y
            J2_y[2] = _perp(r_2).dot(-n_y)
            J2_y = J2_y / J2_y.norm()
        constraints.append((J1_y, J2_y, diff[1]))

    elif isinstance(joint, FixedJoint):
        for i, n in enumerate(
            [torch.tensor([1.0, 0.0], device=device), torch.tensor([0.0, 1.0], device=device)]
        ):
            J1, J2 = torch.zeros(3, device=device), torch.zeros(3, device=device)
            J1[:2] = n
            J1[2] = _perp(r_1).dot(n)
            J1 = J1 / J1.norm()
            if shape_2:
                J2[:2] = -n
                J2[2] = _perp(r_2).dot(-n)
                J2 = J2 / J2.norm()
            constraints.append((J1, J2, diff[i]))

        target_angle_1 = theta_1 + joint.child_target_rotation
        target_angle_2 = theta_2 + joint.parent_target_rotation
        ang_diff = target_angle_2 - target_angle_1
        ang_diff = (ang_diff + torch.pi) % (2 * torch.pi) - torch.pi

        J1_a = torch.tensor([0.0, 0.0, 1.0], device=device)
        J2_a = torch.tensor([0.0, 0.0, -1.0], device=device)
        constraints.append((J1_a, J2_a, ang_diff))

    elif isinstance(joint, PrismaticJoint):
        ang_diff = theta_1 - theta_2
        J1_a = torch.tensor([0.0, 0.0, 1.0], device=device)
        J2_a = torch.tensor([0.0, 0.0, -1.0], device=device)
        constraints.append((J1_a, J2_a, ang_diff))

        axis_world = R_1 @ joint.axis
        perp_axis = _perp(axis_world)
        dist_perp = diff.dot(perp_axis)

        J1_p = torch.zeros(3, device=device)
        J1_p[:2] = perp_axis
        J1_p[2] = diff.dot(axis_world) + _perp(r_1).dot(perp_axis)
        J1_p = J1_p / J1_p.norm()
        J2_p = torch.zeros(3, device=device)
        if shape_2:
            J2_p[:2] = -perp_axis
            J2_p[2] = _perp(r_2).dot(-perp_axis)
            J2_p = J2_p / J2_p.norm()
        constraints.append((J1_p, J2_p, dist_perp))

    return constraints
