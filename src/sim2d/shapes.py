import torch


def shape_to_int(shape):
    shape_to_int_map = {Floor: -1, Circle: 0, Point: 1}
    assert (
        type(shape) in shape_to_int_map.keys()
    ), f"Unknown shape type. Shape type: {type(shape)}, known types: {shape_to_int_map.keys()}"
    return shape_to_int_map[type(shape)]


class Shape:
    def __init__(
        self,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        mass: float,
        restitution: float,
    ):
        self.translation = translation
        self.rotation = rotation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.mass = mass
        self.restitution = restitution


class Floor(Shape):
    def __init__(
        self,
        height: float,
        restitution: float,
    ):
        super().__init__(
            torch.tensor([0.0, height]),
            torch.tensor(0.0),
            torch.tensor([0.0, 0.0]),
            torch.tensor(0.0),
            torch.inf,
            restitution,
        )
        self.height = height


class Circle(Shape):
    def __init__(
        self,
        translation: torch.Tensor,
        velocity: torch.Tensor,
        mass: float,
        restitution: float,
        radius: float,
    ):
        super().__init__(
            translation,
            torch.tensor(0.0),
            velocity,
            torch.tensor(0.0),
            mass,
            restitution,
        )
        self.radius = radius


class Point(Shape):
    def __init__(
        self,
        translation: torch.Tensor,
        velocity: torch.Tensor,
        mass: float,
        restitution: float,
    ):
        super().__init__(
            translation,
            torch.tensor(0.0),
            velocity,
            torch.tensor(0.0),
            mass,
            restitution,
        )
