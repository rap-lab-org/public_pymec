import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.fourier_utils import BasisFunc, get_phik
from jax import vmap


class InfoMap(object):
    def __init__(self, init_p, workspace=jnp.array([[0.0, 1], [0, 1]])) -> None:
        self.dim = len(workspace)
        num_points = int((workspace[:, 1] - workspace[:, 0]).max() * 50)
        # self.domain = jnp.meshgrid(*[jnp.linspace(0, 1, num=num_points)] * self.dim)
        self.domain = jnp.meshgrid(
            *[jnp.linspace(dim[0], dim[1], num=num_points) for dim in workspace]
        )
        self._s = jnp.stack([X.ravel() for X in self.domain]).T
        self.p = init_p
        self.evals = (vmap(self.p)(self._s), self._s)
        self.basis = BasisFunc(n_basis=self.dim * [10], workspace=workspace)
        self.phik = self.update_phik()

    def plot(self, ax=None):
        # only 2-d case is provided
        if self.dim > 2:
            raise NotImplementedError("no 3d scenerio")
        if ax is None:
            plt.contour(
                self.domain[0],
                self.domain[1],
                self.evals[0].reshape(self.domain[0].shape),
                cmap="Greys",
                levels=10,
            )
        else:
            ax.contour(
                self.domain[0],
                self.domain[1],
                self.evals[0].reshape(self.domain[0].shape),
                cmap="Greys",
                levels=10,
            )
        # plt.colorbar()
        # plt.show()

    def update_phi(self, new_p_func):
        self.p = new_p_func
        self.evals = (vmap(self.p)(self._s), self._s)
        self.phik = self.update_phik()

    def update_phik(self):
        phik = get_phik(self.evals, basis=self.basis)
        self.phik = phik
        return phik
