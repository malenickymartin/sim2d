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
        device: torch.device,
        analytical_jac: bool = True,
        beta: float = 0.05,
        eps: float = 1e-6,
        atol: float = 1e-7,
    ):
        self.shapes = shapes
        self.newton_iters = newton_iters
        self.gravity = gravity.to(device)
        self.dt = dt
        self.init_state_fn = init_state_fn
        self.logger = logger
        self.device = device
        self.analytical_jac = analytical_jac
        self.beta = beta
        self.eps = eps
        self.atol = atol

        self.masses = torch.tensor([s.mass for s in self.shapes], device=self.device)
        self.restitutions = torch.tensor([s.restitution for s in self.shapes], device=self.device)

        self.num_shapes = len(shapes)

    def step(self, step_idx: int, state: torch.Tensor, contacts: dict) -> torch.Tensor:
        state_init = state.clone()
        with self.logger.timed_block("initial_guess"):
            state: torch.Tensor = self.init_state_fn(state, contacts, self.dt).clone()
        assert state.shape == self.state_shape(
            contacts
        ), f"State shape is not correct. Expected: {self.state_shape(contacts)}, got {state.shape}."
        with self.logger.timed_block("newton_solve"):
            for i in range(self.newton_iters):
                res_val: torch.Tensor = self.resudial_fn(state, state_init, contacts)
                with self.logger.timed_block("linearization"):
                    if self.analytical_jac:
                        J = self.compute_jacobian(state, state_init, contacts)
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

    def resudial_fn(self, state: torch.Tensor, state_init: torch.Tensor, contacts):
        res = torch.zeros_like(state)
        res[:, 3:] = state[:, 3:]
        res[:, :3] = state[:, :3] - state_init[:, :3] - self.gravity * self.dt
        if contacts["body_idx"].numel() > 0:
            body_idxs = contacts["body_idx"]
            local_idxs = contacts["local_idx"]
            jacobians = contacts["jac"]
            dists = contacts["dist"]

            lambdas = state[body_idxs, 3 + local_idxs]
            force_impulse = -lambdas.unsqueeze(1) * jacobians
            inv_masses = 1.0 / self.masses[body_idxs].unsqueeze(1)
            vel_delta = force_impulse * inv_masses
            res_vel = res[:, :3].clone()
            res_vel.index_add_(0, body_idxs, vel_delta)
            res[:, :3] = res_vel

            b_error = -(self.beta / self.dt) * dists
            b_restitution = self.restitutions[body_idxs].unsqueeze(1) * state_init[body_idxs, :3]
            v_term = state[body_idxs, :3] + b_restitution
            b_scaled = (jacobians * v_term).sum(dim=1)
            a = b_scaled + b_error
            b = lambdas
            fb_vals = self.fischer_burmeister(a, b)
            res[body_idxs, 3 + local_idxs] = fb_vals
        return torch.flatten(res)

    def compute_jacobian(self, state: torch.Tensor, state_init: torch.Tensor, contacts: dict):
        n_shapes, n_vars = state.shape
        total_vars = n_shapes * n_vars
        J = torch.zeros((total_vars, total_vars), device=self.device)
        shape_indices = torch.arange(n_shapes, device=self.device)
        row_starts = shape_indices * n_vars
        col_starts = shape_indices * n_vars
        for k in range(3):
            J[row_starts + k, col_starts + k] = 1.0
        for k in range(3, n_vars):
            diag_idx = row_starts + k
            J[diag_idx, diag_idx] = 1.0
        if contacts["body_idx"].numel() > 0:
            body_idxs = contacts["body_idx"]
            local_idxs = contacts["local_idx"]
            jacobians = contacts["jac"]
            dists = contacts["dist"]
            inv_masses = 1.0 / self.masses[body_idxs]
            lambdas = state[body_idxs, 3 + local_idxs]
            base_rows = body_idxs * n_vars
            base_cols = body_idxs * n_vars
            col_lambda = base_cols + 3 + local_idxs
            for k in range(3):
                J[base_rows + k, col_lambda] = -jacobians[:, k] * inv_masses
            b_error = -(self.beta / self.dt) * dists
            b_restitution = self.restitutions[body_idxs].unsqueeze(1) * state_init[body_idxs, :3]
            v_curr = state[body_idxs, :3]
            a = (jacobians * (v_curr + b_restitution)).sum(dim=1) + b_error
            b = lambdas
            hypot = torch.sqrt(a**2 + b**2 + self.eps)
            dFB_da = 1.0 - a / hypot
            dFB_db = 1.0 - b / hypot
            row_lambda = base_rows + 3 + local_idxs
            for k in range(3):
                J[row_lambda, base_cols + k] = dFB_da * jacobians[:, k]
            J[row_lambda, col_lambda] = dFB_db
        return J

    def fischer_burmeister(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - torch.sqrt(a**2 + b**2 + self.eps)

    def state_shape(self, contacts) -> tuple[int]:
        if contacts["body_idx"].numel() == 0:
            max_contacts = 0
        else:
            max_contacts = int(contacts["counts"].max().item())
        return (self.num_shapes, 3 + max_contacts)
