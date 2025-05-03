import jax.numpy as jnp

from jax import jit, jacfwd, vmap


class DoubleIntegrator:
    def __init__(self, dim=2) -> None:
        self.nx = dim * 2
        self.nu = dim
        # self.u_min = jnp.array([-0.3, -0.3])
        # self.u_max = jnp.array([0.3, 0.3])
        # self.u_min = jnp.array([-0.5, -0.5])
        # self.u_max = jnp.array([0.5, 0.5])
        A = jnp.eye(dim * 2, dim * 2, k=dim)
        # A = np.array([
        #     [0., 0., 1.0, 0.],
        #     [0., 0., 0.0, 1.],
        #     [0., 0., 0.0, 0.],
        #     [0., 0., 0.0, 0.]
        # ])
        B = jnp.eye(dim * 2, dim, -dim)

        # B = np.array([
        #     [0., 0.],
        #     [0., 0.],
        #     [1., 0.],
        #     [0., 1.]
        # ])
        def dxdt(x, u):
            u = jnp.clip(u, -0.5, 0.5)
            # u = jnp.clip(u, self.u_min, self.u_max)
            return A @ x + B @ u

        self.dxdt = jit(dxdt)
        self.getAt = jit(lambda x, u: A)
        self.getBt = jit(lambda x, u: B)


class HomoDynamics:
    def __init__(self, robot_number, dynamics) -> None:
        self.nx = dynamics.nx
        self.nu = dynamics.nu
        self.dynamics = dynamics
        self.Nx = robot_number * self.nx
        self.Nu = robot_number * self.nu

        def dxdt(x, u):
            # x: horizonal, stack
            xi = x.reshape(-1, self.nx)
            ui = u.reshape(-1, self.nu)
            x_dot = vmap(dynamics.dxdt)(x=xi, u=ui)
            return x_dot.flatten()

        self.dxdt = jit(dxdt)
        self.getAt = jit(jacfwd(self.dxdt, argnums=0))
        self.getBt = jit(jacfwd(self.dxdt, argnums=1))
