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

        mass_inertia_tensor = torch.tensor(
            [[s.mass, s.mass, s.inertia] for s in shapes], device=self.device
        )
        self.inv_masses = 1 / mass_inertia_tensor
        self.inv_masses[torch.isinf(self.inv_masses)] = 0.0

        self.num_shapes = len(shapes)

    def step(self, step_idx: int, state: torch.Tensor, constraints: dict) -> torch.Tensor:
        state_init = state.clone()
        with self.logger.timed_block("initial_guess"):
            state: torch.Tensor = self.init_state_fn(state, constraints, self.dt).clone()
            state = self.compute_lambdas_from_velocities(state, state_init, constraints)
        assert state.shape == self.state_shape(
            constraints
        ), f"State shape is not correct. Expected: {self.state_shape(constraints)}, got {state.shape}."
        with self.logger.timed_block("newton_solve"):
            for i in range(self.newton_iters):
                res_val: torch.Tensor = self.residual_fn(state, state_init, constraints)
                if torch.norm(res_val) < self.atol:
                    self.logger.log_engine_data(
                        step_idx, i, state.shape, res_val.detach(), torch.Tensor(), torch.Tensor()
                    )
                    return state.detach()
                with self.logger.timed_block("linearization"):
                    if self.analytical_jac:
                        J = self.compute_jacobian(state, state_init, constraints)
                    else:
                        with torch.enable_grad():
                            state_var = state.detach().requires_grad_(True)
                            J = torch.autograd.functional.jacobian(
                                lambda z: self.residual_fn(z, state_init, constraints), state_var
                            )
                            if J.dim() > 2:
                                J = J.view(J.shape[0], -1)
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

    def compute_lambdas_from_velocities(
        self, state: torch.Tensor, state_init: torch.Tensor, constraints: dict
    ) -> torch.Tensor:
        if constraints["body_idx"].numel() == 0:
            return state
        n_shapes, _ = state.shape
        body_idxs = constraints["body_idx"]
        neighbor_idxs = constraints["neighbor_idx"]
        local_idxs = constraints["local_idx"]
        jacobians = constraints["jac"]
        jacobians_neigh = constraints["jac_neigh"]
        n_constraints = body_idxs.shape[0]

        b = (state[:, :3] - state_init[:, :3] - self.gravity * self.dt).flatten()
        A = torch.zeros(3 * n_shapes, n_constraints, device=self.device)
        col_idxs = torch.arange(n_constraints, device=self.device)
        inv_M = self.inv_masses[body_idxs]
        for k in range(3):
            A[body_idxs * 3 + k, col_idxs] = jacobians[:, k] * inv_M[:, k]

        mask_neigh = neighbor_idxs != -1
        if mask_neigh.any():
            neigh_idxs = neighbor_idxs[mask_neigh]
            inv_M_neigh = self.inv_masses[neigh_idxs]
            col_neigh = col_idxs[mask_neigh]
            for k in range(3):
                A[neigh_idxs * 3 + k, col_neigh] = (
                    jacobians_neigh[mask_neigh, k] * inv_M_neigh[:, k]
                )

        lambdas = torch.linalg.lstsq(A, b).solution

        state = state.clone()
        state[body_idxs, 3 + local_idxs] = lambdas
        return state

    def residual_fn(self, state: torch.Tensor, state_init: torch.Tensor, constraints):
        res = torch.zeros_like(state)
        res[:, 3:] = state[:, 3:]
        res[:, :3] = state[:, :3] - state_init[:, :3] - self.gravity * self.dt
        if constraints["body_idx"].numel() > 0:
            body_idxs = constraints["body_idx"]
            neighbor_idxs = constraints["neighbor_idx"]
            local_idxs = constraints["local_idx"]
            jacobians = constraints["jac"]
            jacobians_neigh = constraints["jac_neigh"]
            dists = constraints["dist"]
            is_equality = constraints["is_equality"]
            restitutions = constraints["restitution"]

            lambdas = state[body_idxs, 3 + local_idxs]
            force_impulse = -lambdas.unsqueeze(1) * jacobians
            inv_M = self.inv_masses[body_idxs]
            vel_delta = force_impulse * inv_M
            res_vel = res[:, :3].clone()
            res_vel.index_add_(0, body_idxs, vel_delta)
            mask_neigh = neighbor_idxs != -1
            if mask_neigh.any():
                neigh_idxs = neighbor_idxs[mask_neigh]
                force_impulse_neigh = (
                    -lambdas[mask_neigh].unsqueeze(1) * jacobians_neigh[mask_neigh]
                )
                inv_M_neigh = self.inv_masses[neigh_idxs]
                vel_delta_neigh = force_impulse_neigh * inv_M_neigh
                res_vel.index_add_(0, neigh_idxs, vel_delta_neigh)
            res[:, :3] = res_vel

            b_error = -(self.beta / self.dt) * dists
            b_restitution = restitutions.unsqueeze(1) * state_init[body_idxs, :3]
            v_term = state[body_idxs, :3] + b_restitution
            b_scaled = (jacobians * v_term).sum(dim=1)
            if mask_neigh.any():
                neigh_b_restitution = (
                    restitutions[mask_neigh].unsqueeze(1) * state_init[neigh_idxs, :3]
                )
                v_term_neigh = state[neigh_idxs, :3] + neigh_b_restitution
                b_scaled[mask_neigh] += (jacobians_neigh[mask_neigh] * v_term_neigh).sum(dim=1)
            a = b_scaled + b_error
            b = lambdas
            fb_vals = self.fischer_burmeister(a, b)
            final_vals = torch.where(is_equality, a, fb_vals)
            res[body_idxs, 3 + local_idxs] = final_vals
        return torch.flatten(res)

    def compute_jacobian(self, state: torch.Tensor, state_init: torch.Tensor, constraints: dict):
        n_shapes, n_vars = state.shape
        total_vars = n_shapes * n_vars
        J = torch.zeros((total_vars, total_vars), device=self.device)
        rows = torch.arange(n_shapes, device=self.device) * n_vars
        for k in range(3):
            J[rows + k, rows + k] = 1.0
        for k in range(3, n_vars):
            J[rows + k, rows + k] = 1.0
        if constraints["body_idx"].numel() > 0:
            body_idxs = constraints["body_idx"]
            neighbor_idxs = constraints["neighbor_idx"]
            local_idxs = constraints["local_idx"]
            jacobians = constraints["jac"]
            jacobians_neigh = constraints["jac_neigh"]
            dists = constraints["dist"]
            is_equality = constraints["is_equality"]
            restitutions = constraints["restitution"]

            inv_M = self.inv_masses[body_idxs]
            lambdas = state[body_idxs, 3 + local_idxs]
            base_rows = body_idxs * n_vars
            col_lambda = base_rows + 3 + local_idxs
            for k in range(3):
                J[base_rows + k, col_lambda] = -jacobians[:, k] * inv_M[:, k]
            mask_neigh = neighbor_idxs != -1
            if mask_neigh.any():
                neigh_idxs = neighbor_idxs[mask_neigh]
                jac_neigh_subset = jacobians_neigh[mask_neigh]
                inv_M_neigh = self.inv_masses[neigh_idxs]
                rows_neigh_vel = neigh_idxs * n_vars
                cols_lambda_subset = col_lambda[mask_neigh]
                for k in range(3):
                    J[rows_neigh_vel + k, cols_lambda_subset] = (
                        -jac_neigh_subset[:, k] * inv_M_neigh[:, k]
                    )
            b_error = -(self.beta / self.dt) * dists
            b_restitution = restitutions.unsqueeze(1) * state_init[body_idxs, :3]
            v_curr = state[body_idxs, :3]
            a = (jacobians * (v_curr + b_restitution)).sum(dim=1)
            if mask_neigh.any():
                neigh_b_restitution = (
                    restitutions[mask_neigh].unsqueeze(1) * state_init[neigh_idxs, :3]
                )
                v_neigh = state[neigh_idxs, :3]
                a[mask_neigh] += (
                    jacobians_neigh[mask_neigh] * (v_neigh + neigh_b_restitution)
                ).sum(dim=1)
            a += b_error
            b = lambdas
            hypot = torch.sqrt(a**2 + b**2 + self.eps)
            d_da = 1.0 - a / hypot
            d_db = 1.0 - b / hypot
            d_da = torch.where(is_equality, torch.ones_like(a), d_da)
            d_db = torch.where(is_equality, torch.zeros_like(b), d_db)
            row_lambda = base_rows + 3 + local_idxs
            for k in range(3):
                J[row_lambda, base_rows + k] = d_da * jacobians[:, k]
            if mask_neigh.any():
                rows_neigh = row_lambda[mask_neigh]
                cols_neigh_base = neigh_idxs * n_vars
                jac_n = jacobians_neigh[mask_neigh]
                dda_n = d_da[mask_neigh]
                for k in range(3):
                    J[rows_neigh, cols_neigh_base + k] = dda_n * jac_n[:, k]
            J[row_lambda, col_lambda] = d_db
        return J

    def fischer_burmeister(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b - torch.sqrt(a**2 + b**2 + self.eps)

    def state_shape(self, constraints) -> tuple[int]:
        if constraints["body_idx"].numel() == 0:
            max_constraints = 0
        else:
            max_constraints = int(constraints["counts"].max().item())
        return (self.num_shapes, 3 + max_constraints)
