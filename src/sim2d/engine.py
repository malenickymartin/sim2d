from collections.abc import Callable

import torch

from .shapes import Shape
from .logger import EngineLogger


class EulerSolver:
    def __init__(
        self,
        shapes: list[Shape],
        newton_iters: int,
        gravity: torch.Tensor,
        dt: float,
        init_state_fn: Callable,
        logger: EngineLogger,
        beta: float = 0.05,
        eps: float = 1e-6,
        atol: float = 1e-7,
    ):
        self.shapes = shapes
        self.newton_iters = newton_iters
        self.gravity = gravity
        self.dt = dt
        self.init_state_fn = init_state_fn
        self.logger = logger
        self.eps = eps
        self.beta = beta
        self.atol = atol

        self.num_shapes = len(shapes)

    def step(self, state: torch.Tensor, contacts: list[list[torch.Tensor]]) -> torch.Tensor:
        # state = num_objects x (velocity_x, velocity_y, angular_velocity, lambdas...)
        # contacts = num_objects x num_contacts_for_object*(dist, contact_jacobian_x,y,alpha)
        state_init = state.clone()
        with self.logger.timed_block("initial_guess"):
            state: torch.Tensor = self.init_state_fn(state, contacts, self.dt).clone()
        assert state.shape == self.state_shape(
            contacts
        ), f"State shape is not correct. Expected: {self.state_shape(contacts)}, got {state.shape}."
        with self.logger.timed_block("newton_solve"):
            for i in range(self.newton_iters):
                with self.logger.timed_block("linearization"):
                    with torch.enable_grad():
                        state_var = state.detach().requires_grad_(True)
                        res_val: torch.Tensor = self.resudial_fn(state_var, state_init, contacts)
                        J = torch.autograd.functional.jacobian(
                            lambda z: self.resudial_fn(z, state_init, contacts), state_var
                        )
                        if J.dim() > 2:
                            J = J.view(J.shape[0], -1)

                if torch.norm(res_val) < self.atol:
                    return state.detach()
                with self.logger.timed_block("linear_solve"):
                    try:
                        delta = torch.linalg.solve(J, -res_val.detach())
                    except RuntimeError:
                        print("solve failed, using lstsq")
                        delta = torch.linalg.lstsq(J, -res_val.detach()).solution
                state = state + torch.reshape(
                    delta, (self.num_shapes, len(delta) // self.num_shapes)
                )
                if torch.norm(delta) < self.atol:
                    return state.detach()
        return state.detach()

    def resudial_fn(self, state, state_init, contacts):
        res = torch.zeros_like(state)
        for i in range(self.num_shapes):
            res[i, :3] += state[i, :3] - state_init[i, :3] - self.gravity * self.dt
            if len(contacts[i]) > 0:
                for j in range(len(contacts[i])):
                    lambda_distributed = contacts[i][j][1] * state[i, 3 + j]
                    res[i, :3] += -lambda_distributed / self.shapes[i].mass
                    b_error = -(self.beta / self.dt) * contacts[i][j][0]
                    b_restitution = self.shapes[i].restitution * state_init[i, :3]
                    b_scaled = torch.dot(contacts[i][j][1], state[i, :3] + b_restitution)
                    res[i, 3 + j] += self.fischer_burmeister(b_scaled + b_error, state[i, 3 + j])
            else:
                res[i, 3:] = state[i, 3:]
        res = torch.flatten(res)
        return res

    def fischer_burmeister(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - torch.sqrt(a**2 + b**2 + self.eps)

    def state_shape(self, contacts) -> tuple[int]:
        return (self.num_shapes, 3 + max([len(a) for a in contacts]))
