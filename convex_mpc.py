# %% Imports
import equinox as eqx
import jax
import jax.numpy as jnp
import klax
import matplotlib.pyplot as plt
from dynax import ISPHS, LyapunovNN
from jax import random as jr
from jaxopt import ProjectedGradient
from jaxtyping import Array
from tqdm.auto import trange

jax.config.update("jax_enable_x64", False)

# %% Parameters

# MPC horizon: how many future steps to optimize over
horizon = 10
# Time step size for integration
dt = 0.01
# Initial state of the system (4D state vector initialized to zero)
x0 = jnp.zeros(4)
# Desired target energy level (for energy regulation objective)
H_desired = 0.025
# Total simulation time
t_final = 4
# Total number of integration steps for the simulation
N_steps = int(t_final / dt)
# Initial guess for control input trajectory (warm start for optimizer)
u_init = 0.01 * jnp.ones((horizon, 1))
# MPC cost weights:
# Q: weight on state deviation from target
Q = 1e4
# R: weight on control effort (penalizes large inputs)
R = 0.1
# Number of outer optimization iterations for the SQP solver
iter_outer = 10
# Small regularization term for numerical stability
eps_reg = 1e-6
# State constraints (upper/lower bounds for each state dimension)
x_bound = jnp.array([0.25, 0.25, 0.3, 0.3])
# Control input saturation limit (absolute bound)
u_bound = 2.0

# %% Neural Network Dynamics Model


def get_deriv_nn_model():
    # Size of the state vector (4 states for the oscillator system)
    state_size = 4

    # Initialize random keys for reproducibility (JAX-style random number generation)
    key = jr.key(0)
    key_ficnn, key_poisson, key_resistive, key_input = jr.split(key, 4)

    # Define a fully input convex neural network (FICNN) to model the scalar Hamiltonian
    ficnn = klax.nn.FICNN(
        in_size=state_size,  # input: state vector
        out_size="scalar",  # output: scalar Hamiltonian value
        width_sizes=[16, 16],  # hidden layers with 16 neurons each
        key=key_ficnn,
    )

    # Wrap FICNN in a LyapunovNN object to interpret output as a valid Hamiltonian
    hamiltonian = LyapunovNN(ficnn, state_size=state_size, key=key_ficnn)

    # Learnable skew-symmetric matrix representing energy-conserving dynamics (structure matrix)
    poisson_matrix = klax.nn.ConstantSkewSymmetricMatrix(state_size, key=key_poisson)

    # Learnable symmetric positive-definite matrix representing dissipative effects (damping)
    resistive_matrix = klax.nn.ConstantSPDMatrix(
        state_size, epsilon=0.0, key=key_resistive
    )

    # Learnable constant matrix defining how control input enters the system
    input_matrix = klax.nn.ConstantMatrix((state_size, 1), key=key_input)

    # Combine all components into an ISPHS model (Input-State Port-Hamiltonian System)
    deriv_model = ISPHS(hamiltonian, poisson_matrix, resistive_matrix, input_matrix)

    # Attempt to load pretrained weights for the model
    try:
        deriv_model = eqx.tree_deserialise_leaves(
            "two_mass_oscillator_model.eqx", deriv_model
        )
    except FileNotFoundError as e:
        raise RuntimeError("Missing model file: two_mass_oscillator_model.eqx") from e

    # Finalize the model (e.g., freeze constants, compile structure)
    finalized_model = klax.finalize(deriv_model)

    # JIT-compile the model for efficient evaluation
    jitted_model = eqx.filter_jit(finalized_model)

    # Additional helper: JIT-compiled Hamiltonian function for direct energy evaluation
    hami_fun = jax.jit(lambda x: finalized_model.hamiltonian(x))

    # Return both the dynamics model and the Hamiltonian function
    return jitted_model, hami_fun


# %% Derivative Class for Linear Two-Mass Oscillator
class Derivative(eqx.Module):
    """
    Derivative function of a linear two mass oscillator system, with a
    force input u acting on the second mass.
    """

    A: Array
    B: Array
    J: Array
    R: Array
    Q: Array

    def __init__(self, m1, m2, k1, k2, d1, d2):
        zeros = jnp.zeros((2, 2))

        # Structure matrix
        mass = jnp.array([[m1, 0], [0, m2]])
        mass_inv = jnp.linalg.inv(mass)
        J = jnp.block([[zeros, mass_inv], [-mass_inv, zeros]])
        self.J = J

        # Resistive matrix
        diss = jnp.array(
            [
                [(d1 + d2) / (m1 * m1), -d2 / (m1 * m2)],
                [-d2 / (m1 * m2), d2 / (m1 * m2)],
            ]
        )
        R = jnp.block([[zeros, zeros], [zeros, diss]])
        self.R = R

        # Hamililtonian quadratic form H=0.5xQx
        Q = jnp.array(
            [
                [k1 + k2, -k2, 0, 0],
                [-k2, k2, 0, 0],
                [0, 0, m1, 0],
                [0, 0, 0, m2],
            ]
        )
        self.Q = Q

        self.A = (J - R) @ Q

        # Input matrix
        self.B = jnp.array([0, 0, 0, 1 / m2])[:, None]

    def __call__(self, t, y, u):
        return self.A @ y + self.B @ u

    def get_hamiltonian(self, y):
        """
        Returns the Hamiltonian H(x) = 0.5 * x^T Q x
        """
        return 0.5 * jnp.inner(y, self.Q @ y)


def make_rk4_step(f):
    """Create a single-step RK4 integrator for dynamics f(t, x, u)."""

    @jax.jit  # JIT-compile for speed
    def _rk4_step(dt, x, u, t=0.0):
        # Classical 4th-order Runge-Kutta integration scheme
        k1 = f(t, x, u)
        k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1, u)
        k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2, u)
        k4 = f(t + dt, x + dt * k3, u)

        # Weighted average of slopes to compute next state
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return _rk4_step


# Instantiate the neural network dynamics model and energy function
dynamics, energy_fn = get_deriv_nn_model()

# Create an RK4 integration step specifically for the NN dynamics
rk4_step_nn = make_rk4_step(dynamics)


# %% MPC Solver Class
class MPCSolver:
    def __init__(
        self,
        *,
        horizon=20,
        dt=1e-2,
        Q=1e4,
        R=1e-1,
        x_min=-jnp.array([0.2, 0.2, 0.3, 0.3]),
        x_max=jnp.array([0.2, 0.2, 0.3, 0.3]),
        u_min=-2.0,
        u_max=2.0,
        penalty=1e6,
        maxiter=100,
        tol=1e-6,
    ):
        self.N = horizon
        self.m = 1
        self.dt = dt
        self.Q = Q
        self.R = R
        self.x_min = x_min
        self.x_max = x_max
        self.u_min = u_min
        self.u_max = u_max
        self.penalty = penalty

        # ----- rollout -----
        @jax.jit  # JIT-compile for speed
        def forward_sim(u_seq, x0):
            def body(x, u):
                x_next = rk4_step_nn(self.dt, x, u)
                return x_next, x_next

            _, xs = jax.lax.scan(body, x0, u_seq)
            return jnp.vstack([x0, xs])  # (N+1, n)

        self._forward_sim = jax.jit(forward_sim)
        self._energy_fn_vec = jax.jit(jax.vmap(energy_fn))

        # ----- cost -----
        def cost_fn(u_seq, x0, H_ref):
            # allow scalar H_ref: broadcast to horizon length
            H_ref_b = jnp.broadcast_to(H_ref, (self.N,))
            x_seq = self._forward_sim(u_seq, x0)
            e = self._energy_fn_vec(x_seq[:-1]) - H_ref_b
            cost = self.Q * jnp.sum(e**2) + self.R * jnp.sum(u_seq**2)
            viol = jnp.where(x_seq[:-1] > self.x_max, x_seq[:-1] - self.x_max, 0.0)
            viol += jnp.where(x_seq[:-1] < self.x_min, self.x_min - x_seq[:-1], 0.0)
            return cost + self.penalty * jnp.sum(viol**2)

        @jax.jit  # JIT-compile for speed
        def proj(u, *args):
            return jnp.clip(u, self.u_min, self.u_max)

        # ----- build the optimizer once -----
        self._pg = ProjectedGradient(
            fun=jax.value_and_grad(cost_fn),  # returns (value, grad)
            projection=proj,
            value_and_grad=True,
            stepsize=1e-2,
            maxiter=maxiter,
            tol=tol,
            jit=True,
        )

    @eqx.filter_jit
    def step(self, x0, H_ref, u_init):
        """
        Solves the MPC optimization for the current state.

        Returns:
            Tuple:
                - First optimal control input (shape: (1,))
                - Full optimal control input sequence (shape: (horizon, 1))
        """
        u_init = jnp.clip(u_init, self.u_min, self.u_max)
        sol = self._pg.run(u_init, None, x0, H_ref)
        # sol.params is the input trajectory, we only need the first control input
        # as the optimal control action at the current state x0
        # sol.state contains the optimization state (e.g., number of iterations)
        return sol.params[0], sol.params


# %% Single MPC solve test (for NN system, use convex MPC)
mpc = MPCSolver(horizon=horizon, dt=dt, Q=Q, R=R, x_min=-x_bound, x_max=x_bound)
u0_opt_nn, _ = mpc.step(
    x0=x0,
    H_ref=H_desired,
    u_init=u_init,
)
print(f"Single Solve Test (NN/convex) for x0 = {x0} returned {u0_opt_nn}")

# %% Closed-loop sim
# Define the *true* physical system parameters (two-mass oscillator)
deriv_true = Derivative(m1=1, m2=2, k1=5, k2=2, d1=0.1, d2=0.1)

# Create an RK4 integration step for the true system
rk4_step_true = make_rk4_step(deriv_true)


def sim_closed_loop(
    mpc: MPCSolver,
    x0: jnp.ndarray,
    u_init: jnp.ndarray,
    N_steps: int,
    *,
    H_desired,
    dt,
    true_sys=False,
):
    """Run a closed-loop MPC simulation using either:
    - True dynamics (`true_sys=True`), or
    - Learned NN surrogate dynamics (`true_sys=False`).
    """

    x = x0  # Current state
    u_prev = u_init  # Previous input sequence (warm start)
    traj = []  # Store trajectory history
    u_hist = []  # Store input history

    for _ in trange(N_steps):
        # At every step, solve MPC for current state
        u0, u_prev = mpc.step(
            x0=x,
            H_ref=H_desired,
            u_init=u_prev,  # Warm-start from previous solve
        )

        # Apply u0 to chosen system (true or learned NN)
        if true_sys:
            # Integrate true physical system using RK4 step for fair comparison
            x = rk4_step_true(dt, x, u0)
        else:
            # Integrate learned NN dynamics using RK4 step
            x = rk4_step_nn(dt, x, u0)

        traj.append(x)
        u_hist.append(u0)

    # Return entire trajectory and input history as arrays
    return jnp.stack(traj), jnp.stack(u_hist)


traj_nn, u_hist_nn = sim_closed_loop(
    mpc=mpc, x0=x0, u_init=u_init, N_steps=N_steps, H_desired=H_desired, dt=dt
)

# --- Run closed-loop simulation using true system dynamics ---
traj_true, u_hist_true = sim_closed_loop(
    mpc=mpc,
    x0=x0,
    u_init=u_init,
    N_steps=N_steps,
    H_desired=H_desired,
    dt=dt,
    true_sys=True,  # Use true physical model
)


# %% Plotting
def plot_closed_loop_results(
    tgrid,
    traj_true,
    traj_nn,
    u_hist_true,
    u_hist_nn,
    x_bound,
    H_desired,
    energy_fn,
    true_system=None,
):
    """Plot diagnostics comparing true vs. NN closed-loop trajectories."""

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(12, 8))

    # --- Plot state trajectories x1 and x2
    for i in range(2):
        ax[0, 0].plot(tgrid, traj_true[:, i], label=f"x{i}_true")
        ax[0, 0].plot(tgrid, traj_nn[:, i], label=f"x{i}_nn", linestyle="--")
        ax[0, 0].hlines(
            [x_bound[i], -x_bound[i]],
            tgrid[0],
            tgrid[-1],
            colors="red",
            linestyles="--",
        )
    ax[0, 0].set_ylabel("x1/x2")
    ax[0, 0].legend()

    # --- Plot state trajectories x3 and x4
    for i in range(2):
        idx = 2 + i
        ax[1, 0].plot(tgrid, traj_true[:, idx], label=f"x{idx}_true")
        ax[1, 0].plot(tgrid, traj_nn[:, idx], label=f"x{idx}_nn", linestyle="--")
        ax[1, 0].hlines(
            [x_bound[idx], -x_bound[idx]],
            tgrid[0],
            tgrid[-1],
            colors="red",
            linestyles="--",
        )
    ax[1, 0].set_ylabel("x3/x4")
    ax[1, 0].legend()

    # --- Plot control inputs
    ax[2, 0].step(tgrid, u_hist_true.squeeze(), where="post", label="u_true")
    ax[2, 0].step(
        tgrid, u_hist_nn.squeeze(), where="post", label="u_nn", linestyle="--"
    )
    ax[2, 0].set_ylabel("u")
    ax[2, 0].legend()

    # --- Plot Hamiltonian (energy) trajectory for NN and true simulation
    H_vals_nn = jax.vmap(energy_fn)(traj_nn)
    ax[0, 1].plot(tgrid, H_vals_nn, label="H_nn")
    if true_system is not None and hasattr(true_system, "get_hamiltonian"):
        H_vals_true = jax.vmap(true_system.get_hamiltonian)(traj_true)
        ax[0, 1].plot(tgrid, H_vals_true, label="H_true")
    ax[0, 1].axhline(H_desired, ls="--", label="H_ref")
    ax[0, 1].set_ylabel("H")
    ax[0, 1].legend()

    # --- Plot trajectory error: ‖x_true − x_nn‖₂
    err_x = jnp.linalg.norm(traj_true - traj_nn, axis=1)
    ax[1, 1].plot(tgrid, err_x)
    ax[1, 1].set_ylabel("‖x_true − x_nn‖₂")

    # --- Plot input difference: |u_true − u_nn|
    err_u = jnp.abs(u_hist_nn - u_hist_true).squeeze()
    ax[2, 1].step(tgrid, err_u, where="post")
    ax[2, 1].set_ylabel("|u_nn - u_true|")
    ax[2, 1].set_xlabel("time [s]")

    fig.tight_layout()
    return fig, ax


# --- Prepare time grid for plotting ---
tgrid = dt * jnp.arange(traj_nn.shape[0])

# --- Generate plots comparing closed-loop trajectories ---
plot_closed_loop_results(
    tgrid,
    traj_true,
    traj_nn,
    u_hist_true,
    u_hist_nn,
    x_bound,
    H_desired,
    energy_fn,
    deriv_true,
)
# %%
