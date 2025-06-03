from dynax import ISPHS, ODESolver, LyapunovNN, aprbs
import klax

import equinox as eqx
import jax.numpy as jnp
from jax import random as jr


def get_model():
    state_size = 4
    model_key = jr.key(0)
    ficnn = klax.nn.FICNN(
        in_size=state_size,
        out_size="scalar",
        width_sizes=[16, 16],
        key=model_key,
    )
    hamiltonian = LyapunovNN(ficnn, state_size=state_size, key=model_key)
    poisson_matrix = klax.nn.ConstantSkewSymmetricMatrix(
        state_size,
        key=model_key,
    )
    resistive_matrix = klax.nn.ConstantSPDMatrix(
        state_size,
        epsilon=0.0,
        key=model_key,
    )
    input_matrix = klax.nn.ConstantMatrix(
        (state_size, 1),
        key=model_key,
    )
    deriv_model = ISPHS(hamiltonian, poisson_matrix, resistive_matrix, input_matrix)
    model = ODESolver(deriv_model)

    # Load the model weights
    try:
        model = eqx.tree_deserialise_leaves("two_mass_oscillator_model.eqx", model)
    except FileNotFoundError:
        print("Model weights file not found. Please ensure 'two_mass_oscillator_model.eqx' exists.")

    jitted_model = eqx.filter_jit(klax.finalize(model))

    # Dummy data
    ts = jnp.linspace(0, 10, 100)
    y0 = jnp.zeros((state_size,))
    us = jnp.zeros((100, 1))
    _ = jitted_model(ts, y0, us)  # Call once to compile

    return jitted_model


def main():
    model = get_model()

    # Generate some dummy data to call the model
    test_key = jr.key(0)
    num_timesteps = 500
    state_size = 4
    test_ts = jnp.linspace(0.0, 150.0, num_timesteps)
    u = jr.uniform(test_key, (num_timesteps, 1))
    y0 = jr.uniform(test_key, (state_size,))
    ys_pred = model(test_ts, y0, u)

    print("Predicted states (Array with shape: [timesteps, state_vector]):")
    print(ys_pred)


if __name__ == "__main__":
    main()
