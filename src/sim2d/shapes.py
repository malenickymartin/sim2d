from abc import ABC

import torch


class Shape(ABC):
    def __init__(
        self,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        mass: torch.Tensor,
        restitution: torch.Tensor,
        inertia: torch.Tensor,
    ):
        self.translation = torch.as_tensor(translation)
        self.rotation = torch.as_tensor(rotation)
        self.velocity = torch.as_tensor(velocity)
        self.angular_velocity = torch.as_tensor(angular_velocity)
        self.mass = torch.as_tensor(mass)
        self.restitution = torch.as_tensor(restitution)
        self.inertia = torch.as_tensor(inertia)

    def to(self, device: torch.device):
        self.translation = self.translation.to(device)
        self.rotation = self.rotation.to(device)
        self.velocity = self.velocity.to(device)
        self.angular_velocity = self.angular_velocity.to(device)
        self.mass = self.mass.to(device)
        self.restitution = self.restitution.to(device)
        self.inertia = self.inertia.to(device)


class Floor(Shape):
    def __init__(
        self,
        height: torch.Tensor,
        restitution: torch.Tensor,
    ):
        super().__init__(
            torch.tensor([0.0, height]),
            torch.tensor(0.0),
            torch.tensor([0.0, 0.0]),
            torch.tensor(0.0),
            torch.tensor(torch.inf),
            restitution,
            torch.tensor(torch.inf),
        )
        self.height = torch.as_tensor(height)


class Circle(Shape):
    def __init__(
        self,
        translation: torch.Tensor,
        velocity: torch.Tensor,
        mass: torch.Tensor,
        restitution: torch.Tensor,
        radius: torch.Tensor,
    ):
        super().__init__(
            translation,
            torch.tensor(0.0),
            velocity,
            torch.tensor(0.0),
            mass,
            restitution,
            mass * radius**2 / 2.0,
        )
        self.radius = torch.as_tensor(radius)

    def to(self, device: torch.device):
        self.radius = self.radius.to(device)
        super().to(device)


class Point(Circle):
    def __init__(
        self,
        translation: torch.Tensor,
        velocity: torch.Tensor,
        mass: torch.Tensor,
        restitution: torch.Tensor,
    ):
        super().__init__(
            translation,
            velocity,
            mass,
            restitution,
            radius=torch.tensor(0.0),
        )
        self.inertia = self.mass.clone()


class Rectangle(Shape):
    def __init__(
        self,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        mass: torch.Tensor,
        restitution: torch.Tensor,
        sides: torch.Tensor,
    ):
        super().__init__(
            translation,
            torch.as_tensor(rotation),
            velocity,
            torch.as_tensor(angular_velocity),
            mass,
            restitution,
            mass * (sides[0] ** 2 + sides[1] ** 2) / 12.0,
        )
        self.sides = torch.as_tensor(sides)

    def to(self, device: torch.device):
        self.sides = self.sides.to(device)
        super().to(device)


SHAPE_TO_INT = {Floor: -1, Circle: 0, Point: 1, Rectangle: 2}
INT_TO_SHAPE = {v: k for k, v in SHAPE_TO_INT.items()}


def shape_to_int(shape):
    shape_type = type(shape)
    assert (
        shape_type in SHAPE_TO_INT.keys()
    ), f"Unknown shape type. Shape type: {shape_type}, known types: {SHAPE_TO_INT.keys()}"
    return SHAPE_TO_INT[shape_type]


def int_to_shape(i: int):
    assert (
        i in INT_TO_SHAPE.keys()
    ), f"Unknown shape type. Shape type: {i}, known types: {INT_TO_SHAPE.keys()}"
    return INT_TO_SHAPE[i]
