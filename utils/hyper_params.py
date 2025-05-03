import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.scipy.stats import multivariate_normal as mvn
from functools import partial

from utils.pdf_utils import InfoMap
from utils.tools import emap_2d
from utils.dynamics import DoubleIntegrator, HomoDynamics
from utils.metric_utils import gaussian_function


_robot_number = 4
_tsteps = 300
params = {
    "workspace": jnp.array([[0.0, 3.0], [0.0, 2.5]]),
    "dt": 0.1,
    "tsteps": _tsteps,
    "robot_number": _robot_number,
    "R": np.diag([5e-3, 5e-3] * _robot_number),
    "Q_z": np.diag([1e-4, 1e-4, 1e-3, 1e-3] * _robot_number),
    "R_v": np.diag([5e-3, 5e-3] * _robot_number),
    "avoidance_radius": 0.3,
    "com_radius": 0.5,
    "r_ergodicity": None,
    "r_probability": None,
    "minimum_probability": None,
    "minimum_ergodicity": None,
    "period_num": None,
    "r_avoidance": 1e-1,
    "r_barrierCost": 1e-1,
    "U_max": np.array([0.5, 0.5] * _robot_number),
    "U_min": np.array([-0.5, -0.5] * _robot_number),
}

if _robot_number == 4:
    init_pos = jnp.array([0.5, 0.3, 1.2, 0.3, 1.9, 0.3, 2.6, 0.3])
    assert len(init_pos) == params["robot_number"] * 2
    # _x_traj = jnp.linspace(init_pos, init_pos, params["tsteps"] + 1)
    _x_traj = jnp.linspace(
        init_pos,
        init_pos + jnp.array([0.0, 2.0] * params["robot_number"]),
        params["tsteps"] + 1,
    )
    _x_dot = jnp.vstack(
        [
            ((_x_traj[1:, :] - _x_traj[:-1, :]) / params["dt"]),
            np.zeros(shape=(2 * params["robot_number"])),
        ]
    )
    _init_x_traj = jnp.column_stack(
        [_x_traj.reshape(-1, 2), _x_dot.reshape(-1, 2)]
    ).reshape(-1, _x_traj.shape[1] + _x_dot.shape[1])
    _init_u_traj = np.zeros((params["tsteps"], params["robot_number"] * 2))

    init_state = _init_x_traj[0, :]
    init_sol = {"x": _init_x_traj[:-1], "u": _init_u_traj}
elif _robot_number == 2:
    init_pos = jnp.array([1.2, 0.3, 1.9, 0.3])
    assert len(init_pos) == params["robot_number"] * 2
    _x_traj = jnp.linspace(
        init_pos,
        init_pos + jnp.array([0.0, 2.0] * params["robot_number"]),
        params["tsteps"] + 1,
    )
    _x_dot = jnp.vstack(
        [
            ((_x_traj[1:, :] - _x_traj[:-1, :]) / params["dt"]),
            np.zeros(shape=(2 * params["robot_number"])),
        ]
    )
    _init_x_traj = jnp.column_stack(
        [_x_traj.reshape(-1, 2), _x_dot.reshape(-1, 2)]
    ).reshape(-1, _x_traj.shape[1] + _x_dot.shape[1])
    _init_u_traj = np.zeros((params["tsteps"], params["robot_number"] * 2))

    init_state = _init_x_traj[0, :]
    init_sol = {"x": _init_x_traj[:-1], "u": _init_u_traj}


def make_infomap(type: int = 11):
    if type == 11:
        centers = jnp.array([[1.0, 0.8], [0.5, 1.7], [2.0, 1.7], [2.5, 0.8]])
        covs = [jnp.array([[0.03, 0.0], [0.0, 0.03]])] * 4
        weights = jnp.array([1 / 4, 1 / 4, 1 / 4, 1 / 4])

    @jit
    def pdf(x):
        return sum(
            w * mvn.pdf(x, center, cov)
            for w, cov, center in zip(weights, covs, centers)
        )

    infomap = InfoMap(init_p=pdf, workspace=params["workspace"])
    return infomap


# some function
func_emap = vmap(partial(emap_2d, workspace_bnds=params["workspace"]))
_single_robot = DoubleIntegrator()
dynamics_multi_robot = HomoDynamics(
    robot_number=params["robot_number"], dynamics=_single_robot
)


func_pair = partial(
    gaussian_function,
    # sigmoid_function,
    sigma=params["com_radius"],
)


params["robot_pair"] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]
