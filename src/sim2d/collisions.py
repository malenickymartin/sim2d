import torch

from .shapes import Shape
from .shapes import Floor
from .shapes import Circle
from .shapes import Point
from .shapes import Rectangle


def compute_collision(
    shape_1: Shape, shape_2: Shape, device: torch.device
) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(shape_1, Circle) and isinstance(shape_2, Floor):
        active, collision_distance, J_1, J_2 = circle_floor(shape_1, shape_2)
    elif isinstance(shape_2, Circle) and isinstance(shape_1, Floor):
        active, collision_distance, J_2, J_1 = circle_floor(shape_2, shape_1)
    elif isinstance(shape_1, Circle) and isinstance(shape_2, Circle):
        active, collision_distance, J_1, J_2 = circle_circle(shape_1, shape_2)
    elif isinstance(shape_1, Point) and isinstance(shape_2, Floor):
        active, collision_distance, J_1, J_2 = point_floor(shape_1, shape_2)
    elif isinstance(shape_2, Point) and isinstance(shape_1, Floor):
        active, collision_distance, J_2, J_1 = point_floor(shape_2, shape_1)
    elif isinstance(shape_1, Point) and isinstance(shape_2, Point):
        active, collision_distance, J_1, J_2 = point_point(shape_1, shape_2)
    elif isinstance(shape_1, Point) and isinstance(shape_2, Circle):
        active, collision_distance, J_1, J_2 = point_circle(shape_1, shape_2)
    elif isinstance(shape_2, Point) and isinstance(shape_1, Circle):
        active, collision_distance, J_2, J_1 = point_circle(shape_2, shape_1)
    elif isinstance(shape_1, Rectangle) and isinstance(shape_2, Floor):
        active, collision_distance, J_1, J_2 = rect_floor(shape_1, shape_2)
    elif isinstance(shape_2, Rectangle) and isinstance(shape_1, Floor):
        active, collision_distance, J_2, J_1 = rect_floor(shape_2, shape_1)
    elif isinstance(shape_1, Rectangle) and isinstance(shape_2, Circle):
        active, collision_distance, J_1, J_2 = rect_circle(shape_1, shape_2)
    elif isinstance(shape_2, Rectangle) and isinstance(shape_1, Circle):
        active, collision_distance, J_2, J_1 = rect_circle(shape_2, shape_1)
    elif isinstance(shape_1, Point) and isinstance(shape_2, Rectangle):
        active, collision_distance, J_1, J_2 = point_rect(shape_1, shape_2)
    elif isinstance(shape_2, Point) and isinstance(shape_1, Rectangle):
        active, collision_distance, J_2, J_1 = point_rect(shape_2, shape_1)
    elif isinstance(shape_1, Rectangle) and isinstance(shape_2, Rectangle):
        active, collision_distance, J_1, J_2 = rect_rect(shape_1, shape_2)
    else:
        raise TypeError(
            f"Combination of input types {(type(shape_1), type(shape_2))} not supported."
        )
    collision_distance = torch.as_tensor(collision_distance).to(device)
    J_1, J_2 = torch.as_tensor(J_1).to(device), torch.as_tensor(J_2).to(device)
    assert (
        collision_distance >= 0.0
    ), f"collision distance is not positive, collision_distance = {collision_distance}"
    assert not active or (
        (torch.norm(J_1) - 1.0) < 1e-6 and (torch.norm(J_2) - 1.0) < 1e-6
    ), f"collision normals are not unit lenght, norm(J_1) = {torch.norm(J_1)}, norm(J_2) = {torch.norm(J_2)}"
    return active, collision_distance, J_1, J_2


def circle_floor(shape_1: Circle, shape_2: Floor) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)
    if shape_1.translation[1] - shape_1.radius < shape_2.translation[1]:
        active = True
        collision_distance = shape_2.translation[1] - shape_1.translation[1] + shape_1.radius
        J_1[1] = 1.0
        J_2[1] = -1.0
    return active, collision_distance, J_1, J_2


def circle_circle(shape_1: Circle, shape_2: Circle) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)
    t_diff = shape_1.translation - shape_2.translation
    t_diff_norm = torch.norm(t_diff)
    if t_diff_norm < shape_1.radius + shape_2.radius:
        assert t_diff_norm != 0
        active = True
        collision_distance = shape_1.radius + shape_2.radius - t_diff_norm
        J_1[:2] = t_diff
        J_1 = J_1.to(t_diff_norm.device) / t_diff_norm
        J_2 = -J_1
    return active, collision_distance, J_1, J_2


def point_floor(shape_1: Point, shape_2: Floor) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)
    if shape_1.translation[1] < shape_2.translation[1]:
        active = True
        collision_distance = shape_2.translation[1] - shape_1.translation[1]
        J_1[1] = 1.0
        J_2[1] = -1.0
    return active, collision_distance, J_1, J_2


def point_point(shape_1: Point, shape_2: Point) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)
    return active, collision_distance, J_1, J_2


def point_circle(shape_1: Point, shape_2: Circle) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)
    t_diff = shape_1.translation - shape_2.translation
    t_diff_norm = torch.norm(t_diff)
    if t_diff_norm < shape_2.radius:
        active = True
        collision_distance = shape_2.radius - t_diff_norm
        J_1[:2] = t_diff
        J_1 = J_1.to(t_diff_norm.device) / t_diff_norm
        J_2 = -J_1
    return active, collision_distance, J_1, J_2


def _get_rotation_matrix(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([c, -s, s, c]).view(2, 2)


def _get_local_point(
    point: torch.Tensor, rect_pos: torch.Tensor, rect_rot: torch.Tensor
) -> torch.Tensor:
    rel_pos = point - rect_pos
    R_inv = _get_rotation_matrix(-rect_rot)
    return torch.matmul(R_inv, rel_pos)


def _get_world_vector(vec: torch.Tensor, rect_rot: torch.Tensor) -> torch.Tensor:
    R = _get_rotation_matrix(rect_rot)
    return torch.matmul(R, vec)


def _get_rect_corners(rect: Rectangle) -> torch.Tensor:
    half_w = rect.sides[0] / 2
    half_h = rect.sides[1] / 2
    corners_local = torch.stack(
        [half_w, half_h, -half_w, half_h, -half_w, -half_h, half_w, -half_h]
    ).view(4, 2)
    R = _get_rotation_matrix(rect.rotation)
    corners_world = rect.translation + torch.matmul(corners_local, R.t())
    return corners_world


def _project_rect(corners: torch.Tensor, axis: torch.Tensor) -> tuple[float, float]:
    projections = torch.matmul(corners, axis)
    return torch.min(projections), torch.max(projections)


def _get_support_point(rect: Rectangle, direction: torch.Tensor) -> torch.Tensor:
    R_inv = _get_rotation_matrix(-rect.rotation)
    dir_local = torch.matmul(R_inv, direction)
    half_w = rect.sides[0] / 2
    half_h = rect.sides[1] / 2
    sx = half_w if dir_local[0] > 0 else -half_w
    sy = half_h if dir_local[1] > 0 else -half_h
    p_local = torch.stack([sx, sy])
    R = _get_rotation_matrix(rect.rotation)
    return rect.translation + torch.matmul(R, p_local)


def rect_rect(shape_1: Rectangle, shape_2: Rectangle) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)
    R1 = _get_rotation_matrix(shape_1.rotation)
    R2 = _get_rotation_matrix(shape_2.rotation)
    axes = [R1[:, 0], R1[:, 1], R2[:, 0], R2[:, 1]]
    corners_1 = _get_rect_corners(shape_1)
    corners_2 = _get_rect_corners(shape_2)

    min_overlap = torch.tensor(torch.inf)
    best_axis = torch.zeros(2)
    axis_owner = 0

    for i, axis in enumerate(axes):
        min_1, max_1 = _project_rect(corners_1, axis)
        min_2, max_2 = _project_rect(corners_2, axis)
        if max_1 < min_2 or max_2 < min_1:
            return False, torch.tensor(0.0), J_1, J_2

        overlap = min(max_1, max_2) - max(min_1, min_2)
        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = axis
            axis_owner = 1 if i < 2 else 2
            diff = shape_1.translation - shape_2.translation
            if torch.dot(diff, best_axis) < 0:
                best_axis = -best_axis
    active = True
    collision_distance = min_overlap
    normal = best_axis
    if axis_owner == 1:
        contact_point = _get_support_point(shape_2, normal)
    else:
        contact_point = _get_support_point(shape_1, -normal)

    J_1[:2] = normal
    r1 = contact_point - shape_1.translation
    J_1[2] = r1[0] * normal[1] - r1[1] * normal[0]

    J_2[:2] = -normal
    r2 = contact_point - shape_2.translation
    J_2[2] = r2[0] * (-normal[1]) - r2[1] * (-normal[0])
    J_1 = J_1 / torch.norm(J_1)
    J_2 = J_2 / torch.norm(J_2)

    return active, collision_distance, J_1, J_2


def point_rect(shape_1: Point, shape_2: Rectangle) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)

    p_local = _get_local_point(shape_1.translation, shape_2.translation, shape_2.rotation)
    half_w = shape_2.sides[0] / 2
    half_h = shape_2.sides[1] / 2
    if -half_w <= p_local[0] <= half_w and -half_h <= p_local[1] <= half_h:
        active = True
        dx_pos = half_w - p_local[0]
        dx_neg = p_local[0] + half_w
        dy_pos = half_h - p_local[1]
        dy_neg = p_local[1] + half_h

        min_dist = min(dx_pos, dx_neg, dy_pos, dy_neg)
        collision_distance = min_dist
        n_local = torch.zeros(2)
        if min_dist == dx_pos:
            n_local = torch.tensor([1.0, 0.0])
        elif min_dist == dx_neg:
            n_local = torch.tensor([-1.0, 0.0])
        elif min_dist == dy_pos:
            n_local = torch.tensor([0.0, 1.0])
        else:
            n_local = torch.tensor([0.0, -1.0])
        normal_world = _get_world_vector(n_local, shape_2.rotation)

        J_1[:2] = normal_world

        J_2[:2] = -normal_world
        r = shape_1.translation - shape_2.translation
        f_rect = -normal_world
        J_2[2] = r[0] * f_rect[1] - r[1] * f_rect[0]

        J_1 = J_1 / torch.norm(J_1)
        J_2 = J_2 / torch.norm(J_2)

    return active, collision_distance, J_1, J_2


def rect_floor(shape_1: Rectangle, shape_2: Floor) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)

    lowest_point = _get_support_point(
        shape_1, torch.tensor([0.0, -1.0], device=shape_1.rotation.device)
    )

    if lowest_point[1] < shape_2.translation[1]:
        active = True
        collision_distance = shape_2.translation[1] - lowest_point[1]
        normal = torch.tensor([0.0, 1.0])

        J_1[:2] = normal
        r = lowest_point - shape_1.translation
        J_1[2] = r[0] * normal[1] - r[1] * normal[0]

        J_2[1] = -1.0
        J_1 = J_1 / torch.norm(J_1)
        J_2 = J_2 / torch.norm(J_2)

    return active, collision_distance, J_1, J_2


def rect_circle(shape_1: Rectangle, shape_2: Circle) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, torch.tensor(0.0), torch.zeros(3), torch.zeros(3)

    c_local = _get_local_point(shape_2.translation, shape_1.translation, shape_1.rotation)

    half_w = shape_1.sides[0] / 2
    half_h = shape_1.sides[1] / 2

    closest_local = torch.stack(
        [torch.clamp(c_local[0], -half_w, half_w), torch.clamp(c_local[1], -half_h, half_h)]
    )

    diff = c_local - closest_local
    dist_sq = torch.dot(diff, diff)

    if dist_sq == 0:
        dx_pos = half_w - c_local[0]
        dx_neg = c_local[0] + half_w
        dy_pos = half_h - c_local[1]
        dy_neg = c_local[1] + half_h

        min_dist = min(dx_pos, dx_neg, dy_pos, dy_neg)

        if min_dist == dx_pos:
            normal_local = torch.tensor([1.0, 0.0])
        elif min_dist == dx_neg:
            normal_local = torch.tensor([-1.0, 0.0])
        elif min_dist == dy_pos:
            normal_local = torch.tensor([0.0, 1.0])
        else:
            normal_local = torch.tensor([0.0, -1.0])
        normal_local = normal_local.to(c_local.device)

        collision_distance = shape_2.radius + min_dist
        contact_local = c_local + normal_local * min_dist
        contact_local = c_local

    elif dist_sq < shape_2.radius**2:
        active = True
        dist = torch.sqrt(dist_sq)
        collision_distance = shape_2.radius - dist

        normal_local = diff / dist
        contact_local = closest_local
    else:
        return False, torch.tensor(0.0), J_1, J_2

    active = True

    normal_world = _get_world_vector(normal_local, shape_1.rotation)
    J_1[:2] = -normal_world

    contact_world = shape_1.translation + _get_world_vector(contact_local, shape_1.rotation)
    r = contact_world - shape_1.translation
    f_rect = -normal_world
    J_1[2] = r[0] * f_rect[1] - r[1] * f_rect[0]
    J_2[:2] = normal_world

    J_1 = J_1 / torch.norm(J_1)
    J_2 = J_2 / torch.norm(J_2)

    return active, collision_distance, J_1, J_2
