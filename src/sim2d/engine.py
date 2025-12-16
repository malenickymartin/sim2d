from collections.abc import Callable

import torch

from .shapes import Shape
from .logger import EngineLogger


def compute_jacobian(self, state: torch.Tensor, state_init: torch.Tensor, contacts: list):
    n_shapes, n_vars = state.shape
    total_vars = n_shapes * n_vars
    J = torch.zeros((total_vars, total_vars), device=state.device)
    for i in range(n_shapes):
        row_start = i * n_vars
        col_start = i * n_vars
        J[row_start : row_start + 3, col_start : col_start + 3] = torch.eye(3)
        num_contacts = len(contacts[i])
        if num_contacts > 0:
            inv_mass = 1.0 / self.shapes[i].mass
            for j in range(num_contacts):
                dist, normal = contacts[i][j]
                lambda_val = state[i, 3 + j]
                J[row_start : row_start + 3, col_start + 3 + j] = -normal * inv_mass
                b_error = -(self.beta / self.dt) * dist
                b_restitution = self.shapes[i].restitution * state_init[i, :3]
                a = torch.dot(normal, state[i, :3] + b_restitution) + b_error
                b = lambda_val
                hypot = torch.sqrt(a**2 + b**2 + self.eps)
                dFB_da = 1.0 - a / hypot
                dFB_db = 1.0 - b / hypot
                row_idx = row_start + 3 + j
                J[row_idx, col_start : col_start + 3] = dFB_da * normal
                J[row_idx, col_start + 3 + j] = dFB_db
        unused_start = 3 + num_contacts
        if unused_start < n_vars:
            idx_range = range(row_start + unused_start, row_start + n_vars)
            J[idx_range, idx_range] = 1.0
    return J


class EulerSolver:
    def __init__(
        self,
        shapes: list[Shape],
        newton_iters: int,
        gravity: torch.Tensor,
        dt: float,
        init_state_fn: Callable,
        logger: EngineLogger,
        analytical_jac: bool = True,
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
        self.analytical_jac = analytical_jac
        self.beta = beta
        self.eps = eps
        self.atol = atol

        self.num_shapes = len(shapes)

    def step(
        self, step_idx: int, state: torch.Tensor, contacts: list[list[torch.Tensor]]
    ) -> torch.Tensor:
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
                    res_val: torch.Tensor = self.resudial_fn(state, state_init, contacts)
                    if self.analytical_jac:
                        J = compute_jacobian(self, state, state_init, contacts)
                    else:
                        with torch.enable_grad():
                            state_var = state.detach().requires_grad_(True)
                            J = torch.autograd.functional.jacobian(
                                lambda z: self.resudial_fn(z, state_init, contacts), state_var
                            )
                            if J.dim() > 2:
                                J = J.view(J.shape[0], -1)
                if torch.norm(res_val) < self.atol:
                    self.logger.log_engine_data(
                        step_idx, i, state.shape, res_val.detach(), torch.Tensor(), J
                    )
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
                self.logger.log_engine_data(step_idx, i, state.shape, res_val.detach(), delta, J)
                if torch.norm(delta) < self.atol:
                    return state.detach()
        return state.detach()

    def resudial_fn(self, state: torch.Tensor, state_init: torch.Tensor, contacts: list):
        res = torch.zeros_like(state)
        for i in range(self.num_shapes):
            num_contacts = len(contacts[i])
            res[i, :3] += state[i, :3] - state_init[i, :3] - self.gravity * self.dt
            if num_contacts > 0:
                for j in range(num_contacts):
                    lambda_distributed = contacts[i][j][1] * state[i, 3 + j]
                    res[i, :3] += -lambda_distributed / self.shapes[i].mass
                    b_error = -(self.beta / self.dt) * contacts[i][j][0]
                    b_restitution = self.shapes[i].restitution * state_init[i, :3]
                    b_scaled = torch.dot(contacts[i][j][1], state[i, :3] + b_restitution)
                    res[i, 3 + j] = self.fischer_burmeister(b_scaled + b_error, state[i, 3 + j])
            if num_contacts < (state.shape[1]):
                res[i, 3 + num_contacts :] = state[i, 3 + num_contacts :]
        res = torch.flatten(res)
        return res

    def fischer_burmeister(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - torch.sqrt(a**2 + b**2 + self.eps)

    def state_shape(self, contacts) -> tuple[int]:
        return (self.num_shapes, 3 + max([len(a) for a in contacts]))
