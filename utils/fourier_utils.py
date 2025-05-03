import jax.numpy as jnp
import numpy as np

from functools import partial
from jax import vmap, jit, jacfwd


def get_hk(k):  # normalizing factor for basis function
    _hk = (2.0 * k + np.sin(2 * k)) / (4.0 * k)
    _hk = _hk.at[np.isnan(_hk)].set(1.0)
    return np.sqrt(np.prod(_hk))


def get_ck(trajectory, basis, tf, dt):
    ck = jnp.sum(vmap(basis.fk)(trajectory), axis=0)
    ck = ck / basis.hk_list
    ck = ck * dt / tf
    return ck


def get_ck_avg(trajectory, basis, tf, dt, robot_number, _nx):
    traj_reshape = trajectory.reshape(trajectory.shape[0], -1, 4).transpose(1, 0, 2)[
        :, :, :2
    ]
    ck = vmap(get_ck, in_axes=(0, None, None, None))(traj_reshape, basis, tf, dt)
    return jnp.average(ck, axis=0)


def get_phik(vals, basis):
    _phi, _x = vals
    phik = jnp.dot(_phi, vmap(basis.fk)(_x))
    phik = phik / phik[0]
    phik = phik / basis.hk_list
    return phik


def recon_from_fourier(basis_coef, basis, x_vals, normalize=False):
    phi = jnp.dot(vmap(basis.fk)(x_vals), basis_coef)
    if normalize:
        min_phi = jnp.min(phi)
        phi = phi - min_phi + 0.1
        phi = phi / jnp.sum(phi)
    return phi


class BasisFunc(object):
    def __init__(self, n_basis, workspace=np.array([[0, 1], [0, 1]])) -> None:
        assert len(n_basis) == workspace.shape[0]
        self.dim = len(n_basis)
        kmesh = jnp.meshgrid(*[jnp.arange(0, n_max, step=1) for n_max in n_basis])
        self.k_list = jnp.stack([_k.ravel() for _k in kmesh]).T * jnp.pi
        self.hk_list = jnp.array([get_hk(_k) for _k in self.k_list])
        # Note: the first dim should be the trajectory
        self._fk = lambda k, x: jnp.prod(
            jnp.cos(x[: self.dim] * k / (workspace[:, 1] - workspace[:, 0])), axis=0
        )
        # self.fk_kvmap = vmap(self._fk, in_axes=(0, None))
        # self.fk_xvmap = vmap(self._fk, in_axes=(None, 0))
        self.fk = partial(vmap(self._fk, in_axes=(0, None)), self.k_list)
        self.dfk = jit(jacfwd(self.fk))
        # self.d2fk = jit(jacfwd(self.dfk))
