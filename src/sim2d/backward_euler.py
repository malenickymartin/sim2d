import time

import matplotlib.pyplot as plt
import torch


def fb(a: torch.Tensor, b: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Standard Fischer-Burmeister function for complementarity.
    phi(a, b) = a + b - sqrt(a^2 + b^2 + eps)
    """
    return a + b - torch.sqrt(a**2 + b**2 + epsilon)


def newton_root_finder(residual_fn, x_init, args=(), tol=1e-7, max_iter=30):
    """
    A standalone Newton-Raphson solver using torch.linalg.solve.

    It locally enables gradients *only* to compute the Jacobian of the residual
    function, ensuring the global simulation does not need to track history.
    """
    x = x_init.clone()

    for i in range(max_iter):
        # 1. Enable grad locally just to compute the Jacobian (J)
        with torch.enable_grad():
            x_var = x.detach().requires_grad_(True)
            f_val = residual_fn(x_var, *args)

            # Compute Jacobian: J = d(f_val)/d(x_var)
            # torch.autograd.functional.jacobian is convenient here
            J = torch.autograd.functional.jacobian(lambda z: residual_fn(z, *args), x_var)

            # Flatten J if necessary (handle cases where output shape != input shape,
            # though here they should match for root finding)
            if J.dim() > 2:
                J = J.view(J.shape[0], -1)

        # 2. Check for convergence (Norm of residual)
        if torch.norm(f_val) < tol:
            return x.detach()

        # 3. Solve the linear system: J * delta = -F
        # Using torch.linalg.solve as requested
        try:
            # We detach f_val so we don't track the solve step in a graph
            delta = torch.linalg.solve(J, -f_val.detach())
        except RuntimeError:
            # Fallback to least squares if J is singular (rare in FB formulations but possible)
            delta = torch.linalg.lstsq(J, -f_val.detach()).solution

        # 4. Update guess
        x = x + delta

        # Check convergence (Norm of step size)
        if torch.norm(delta) < tol:
            return x.detach()

    return x.detach()


def backward_euler_contact_step(y_curr, dt, g_param, restitution_coeff_float, collision_active):
    """
    Perform a single backward Euler step with contact handling using local Newton solve.
    Input vars: y_curr [x, y, vx, vy]
    """
    e = restitution_coeff_float

    # Define the residual function for the implicit step
    # z_next_vars = [vx_next, vy_next, lambda_next]
    def residual_contact(z_next_vars, y_c, dt_val, g_val, is_colliding):
        vx_next, vy_next, lambda_next = z_next_vars[0], z_next_vars[1], z_next_vars[2]

        # Physics Residuals (Implicit Euler)
        # 1. vx_next - vx_curr = 0  (Assuming no x-forces)
        res1 = vx_next - y_c[2]

        # 2. vy_next - vy_curr + dt*g - lambda = 0
        # (Note: signs depend on gravity direction config. Kept consistent with provided snippet)
        res2 = vy_next - y_c[3] + dt_val * g_val - lambda_next

        # 3. Complementarity
        if is_colliding:
            # Baumgarte stabilization terms provided in original snippet
            b_err = (0.01 / dt_val) * y_c[1]
            b_rest = e * y_c[3]

            # Fischer-Burmeister: phi(lambda, velocity_constraint) = 0
            val_constraint = vy_next + b_err + b_rest
            res3 = fb(lambda_next, val_constraint)
        else:
            # If not colliding, lambda must be 0
            res3 = lambda_next

        return torch.stack([res1, res2, res3])

    # Initial Guess: Explicit Euler prediction
    vx_guess = y_curr[2]
    vy_guess = y_curr[3] - dt * g_param  # Guess velocity
    z_guess = torch.tensor([vx_guess, vy_guess, 0.0], device=y_curr.device)

    # Solve for [vx_next, vy_next, lambda]
    # We pass 'y_curr' as a constant parameter
    z_next_sol = newton_root_finder(
        residual_contact,
        z_guess,
        args=(y_curr, dt, g_param, collision_active),
        tol=1e-7,
        max_iter=30,
    )

    vx_next_sol = z_next_sol[0]
    vy_next_sol = z_next_sol[1]
    lambda_contact_sol = z_next_sol[2]

    # Clean up lambda if no collision (numerical noise)
    if not collision_active:
        lambda_contact_sol = torch.zeros_like(lambda_contact_sol)

    # Update Positions: x_next = x_curr + dt * vb_next
    x_next_pos = y_curr[0] + dt * vx_next_sol
    y_next_pos = y_curr[1] + dt * vy_next_sol

    y_next_state = torch.stack([x_next_pos, y_next_pos, vx_next_sol, vy_next_sol])

    return y_next_state, lambda_contact_sol


def backward_euler_integrator(
    ts,
    y0,
    integration_params,
    restitution_coeff=0.0,
    collision_threshold=0.0,
):
    """
    Main loop for computing the trajectory.
    """
    g_val = integration_params[0]
    n_steps = len(ts) - 1

    yt_list = [y0.clone()]
    y_curr = y0.clone()

    device = y_curr.device

    # Ensure constants are tensors on the right device
    if not isinstance(g_val, torch.Tensor):
        g_param = torch.tensor(g_val, device=device)
    else:
        g_param = g_val.to(device)

    print(f"Starting simulation: {n_steps} steps...")

    # Run loop without global gradient tracking
    with torch.no_grad():
        for i in range(n_steps):
            t_curr, t_next = ts[i], ts[i + 1]
            dt = t_next - t_curr

            # Determine if collision is potentially active (simple geometric check)
            # In implicit methods, this often turns the constraint "on" in the solver
            is_collision_active = bool(y_curr[1].item() < collision_threshold)

            y_next_state, _ = backward_euler_contact_step(
                y_curr, dt, g_param, restitution_coeff, is_collision_active
            )

            yt_list.append(y_next_state)
            y_curr = y_next_state

    return torch.stack(yt_list, dim=0)


if __name__ == "__main__":
    # --- Configuration ---
    # Gravity (Positive here naturally implies a downward force in the residual equation: vy_next + dt*g ... = vy_curr)
    # Check residual: vy_next - vy_curr + dt * g = lambda. => vy_next = vy_curr - dt*g + lambda.
    # If standard gravity is -9.8, we should likely set g_val = 9.8 if the equation handles signs explicitly,
    # or set g_val = -9.8 and change signs.
    # Based on original script setting g=0.0 for test, we keep that, or set simple gravity.
    g_gravity_val = 9.81
    g_gravity = torch.tensor(g_gravity_val)

    sim_params = [g_gravity]

    # State: [x, y, vx, vy]
    # Starting high up (y=2.0), moving right (vx=1.0), falling down (vy=-5.0)
    initial_y_pos = 2.0
    y_initial_state = torch.tensor([0.0, initial_y_pos, 1.0, -5.0])

    restitution = 0.8  # Bouncy
    sim_time = 3.0
    dt_val = 0.005
    num_steps = int(sim_time / dt_val)
    ts = torch.linspace(0, sim_time, num_steps + 1)

    collision_thresh = 0.0  # Height below which we solve complementarity

    # --- Run Simulation ---
    start_time = time.time()

    states_fixed = backward_euler_integrator(
        ts=ts,
        y0=y_initial_state,
        integration_params=sim_params,
        restitution_coeff=restitution,
        collision_threshold=collision_thresh,
    )

    end_time = time.time()
    print(f"Simulation finished in {end_time - start_time:.4f}s.")
    print(f"Final State: Pos=({states_fixed[-1, 0]:.3f}, {states_fixed[-1, 1]:.3f})")

    # --- Plotting ---
    # Convert to numpy for plotting
    traj = states_fixed.cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(traj[:, 0], traj[:, 1], "r-o", markersize=3, label="Backward Euler")
    plt.scatter(traj[0, 0], traj[0, 1], c="black", marker="x", s=100, label="Start")

    plt.title(f"Rigid Body Contact Simulation \ne={restitution}, dt={dt_val}")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.axhline(0, color="k", linestyle="--", linewidth=1.0, label="Ground")
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")

    # plt.savefig("be_simulation_result.png")
    # print("Saved plot to 'be_simulation_result.png'")
    plt.show()
