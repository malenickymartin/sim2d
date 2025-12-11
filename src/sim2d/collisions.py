import torch

from .shapes import Shape
from .shapes import Floor
from .shapes import Circle
from .shapes import Point


def compute_collision(shape_1: Shape, shape_2: Shape) -> tuple[bool, float, torch.Tensor]:
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
    else:
        raise TypeError(
            f"Combination of input types {(type(shape_1), type(shape_2))} not supported."
        )
    assert (
        collision_distance >= 0.0
    ), f"collision distance is not positive, collision_distance = {collision_distance}"
    assert not active or (
        (torch.norm(J_1) - 1.0) < 1e-6 and (torch.norm(J_2) - 1.0) < 1e-6
    ), f"collision normals are not unit lenght, norm(J_1) = {torch.norm(J_1)}, norm(J_2) = {torch.norm(J_2)}"
    return active, collision_distance, J_1, J_2


def circle_floor(shape_1: Circle, shape_2: Floor) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, 0.0, torch.zeros(3), torch.zeros(3)
    if shape_1.translation[1] - shape_1.radius < shape_2.translation[1]:
        active = True
        collision_distance = shape_2.translation[1] - shape_1.translation[1] + shape_1.radius
        J_1[1] = 1.0
        J_2[1] = -1.0
    return active, collision_distance, J_1, J_2


def circle_circle(shape_1: Circle, shape_2: Circle) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, 0.0, torch.zeros(3), torch.zeros(3)
    t_diff = shape_1.translation - shape_2.translation
    t_diff_norm = torch.norm(t_diff)
    if t_diff_norm < shape_1.radius + shape_2.radius:
        assert t_diff_norm != 0
        active = True
        collision_distance = shape_1.radius + shape_2.radius - t_diff_norm
        J_1[:2] = t_diff
        J_1 = J_1 / t_diff_norm
        J_2 = -J_1
    return active, collision_distance, J_1, J_2


def point_floor(shape_1: Point, shape_2: Floor) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, 0.0, torch.zeros(3), torch.zeros(3)
    if shape_1.translation[1] < shape_2.translation[1]:
        active = True
        collision_distance = shape_2.translation[1] - shape_1.translation[1]
        J_1[1] = 1.0
        J_2[1] = -1.0
    return active, collision_distance, J_1, J_2


def point_point(shape_1: Point, shape_2: Point) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, 0.0, torch.zeros(3), torch.zeros(3)
    return active, collision_distance, J_1, J_2


def point_circle(shape_1: Point, shape_2: Circle) -> tuple[bool, float, torch.Tensor]:
    active, collision_distance, J_1, J_2 = False, 0.0, torch.zeros(3), torch.zeros(3)
    t_diff = shape_1.translation - shape_2.translation
    t_diff_norm = torch.norm(t_diff)
    if t_diff_norm < shape_2.radius:
        active = True
        collision_distance = shape_2.radius - t_diff_norm
        J_1[:2] = t_diff
        J_1 = J_1 / t_diff_norm
        J_2 = -J_1
    return active, collision_distance, J_1, J_2
