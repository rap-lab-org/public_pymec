import jax.numpy as jnp
from jax import vmap
from scipy.special import comb


def sample_points(evals, num_samples, rng):
    idx = rng.choice(jnp.arange(evals[1].shape[0]), size=num_samples, p=evals[0])
    return evals[1][idx]


def emap_2d(x, workspace_bnds):
    """Function that maps states to workspace"""
    return jnp.array(
        [
            (x[0] - workspace_bnds[0][0])
            / (workspace_bnds[0][1] - workspace_bnds[0][0]),
            (x[1] - workspace_bnds[1][0])
            / (workspace_bnds[1][1] - workspace_bnds[1][0]),
        ]
    )


def euclidean_distance(point1, point2):
    return jnp.sqrt(jnp.sum((point1 - point2) ** 2, axis=-1))


def compute_distances(time_step_positions, dim=3, nx=2):
    # 假设 positions 是一个形状为 (T, robot_num*3) 的数组
    # (T, robot_num*3) --> (T, robot_num, 3)
    tmp = time_step_positions.reshape(time_step_positions.shape[0], -1, nx)
    # 进行vmap (盘算时间步中的距离)
    # 返回 (robot_num, robot_num, T)
    return vmap(euclidean_distance, in_axes=0, out_axes=2)(
        tmp[:, :, jnp.newaxis, :dim], tmp[:, jnp.newaxis, :, :dim]
    )


# get vec based on mat and index_pair
def get_vec(mat, index_pair):
    # return jnp.vstack([mat[i, j] for i, j in index_pair])
    return jnp.column_stack([mat[i, j] for i, j in index_pair])


def get_uppermat(mat, robot_num=3):
    list = []
    for i in range(robot_num):
        for j in range(i + 1, robot_num):
            list.append(mat[i, j])
    return jnp.vstack(list)


# some tools
def barrier_cost(e):
    """Barrier function to avoid robot going out of workspace"""
    return (jnp.maximum(0, e - 1) + jnp.maximum(0, -e)) ** 2


def avoidance_constraints(xt, robot_number, _nx, collision_epsilon=0.4):
    _avoidance_box = []
    for i in range(robot_number - 1):
        _xi = xt[:, i * _nx : i * _nx + 2]
        for j in range(i + 1, robot_number):
            _xj = xt[:, j * _nx : j * _nx + 2]
            _avoidance_box.append(
                (-jnp.linalg.norm((_xi - _xj), axis=1) ** 2 + collision_epsilon**2)
                / comb(robot_number, 2)
            )

    return jnp.array(_avoidance_box).flatten()
