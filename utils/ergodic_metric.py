import jax.numpy as jnp


class ErgodicMetric(object):
    def __init__(self, basis) -> None:
        self.basis = basis
        self.lamk = (1.0 + jnp.linalg.norm(basis.k_list / jnp.pi, axis=1) ** 2) ** (
            -(basis.dim + 1) / 2.0
        )

    def __call__(self, ck, phik):
        return jnp.sum(self.lamk * (ck - phik) ** 2)
