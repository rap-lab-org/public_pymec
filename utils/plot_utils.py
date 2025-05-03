# %%
import jax.numpy as jnp
from jax import vmap
import numpy as np
import matplotlib.pyplot as plt
from utils.hyper_params import params
from matplotlib.colors import to_hex
from matplotlib import cm

# %%
# plot settings
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{xcolor,amsmath,amsfonts}"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.major.width"] = 1.0
plt.rcParams["ytick.major.width"] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

COLORS = [to_hex(cm.tab10(i)) for i in range(10)]
MARKERS = [
    "o",  # circle
    "^",  # triangle up
    "s",  # square
    "D",  # diamond
    "v",  # triangle down
    "P",  # plus (filled)
    "X",  # x (filled)
    "h",  # hexagon
]
LINE_STYLES = ["-", "--", "-.", ":"]
LINE_WIDTH = 2.0


# %%
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


# %%
period_number = 3
robot_number = 4
dynamics_dim = 4
OBS_radius = np.array([0.3, 0.3])
OBS_center = np.array([[1.3, 1.5], [1.8, 0.7]])
OBS_rect = np.array([0.3, 0.2])  # width, height


# %%
def plot_traj_period(traj, no_period, ax):
    labels = ["Robot 1", "Robot 2", "Robot 3", "Robot 4"]
    for robot_id in range(robot_number):
        # 主轨迹
        ax.plot(
            traj[no_period][:, 4 * robot_id],
            traj[no_period][:, 4 * robot_id + 1],
            color=COLORS[robot_id],
            linestyle=LINE_STYLES[0],
            linewidth=LINE_WIDTH,
            label=labels[robot_id] if no_period == 0 else None,
            zorder=2,
        )
        # 起点
        ax.scatter(
            traj[no_period][0, 4 * robot_id],
            traj[no_period][0, 4 * robot_id + 1],
            color=COLORS[robot_id],
            marker="o",
            s=100,
            alpha=0.8,
            zorder=3,
            edgecolors="white",
            linewidth=1,
        )
        # 终点
        ax.scatter(
            traj[no_period][-1, 4 * robot_id],
            traj[no_period][-1, 4 * robot_id + 1],
            color=COLORS[robot_id],
            marker="*",
            s=150,
            alpha=0.8,
            zorder=3,
            edgecolors="white",
            linewidth=1,
        )

        # 前后阶段的轨迹连接
        if no_period > 0:
            ax.plot(
                traj[no_period - 1][-5:, 4 * robot_id],
                traj[no_period - 1][-5:, 4 * robot_id + 1],
                color=COLORS[robot_id],
                linestyle=LINE_STYLES[1],
                linewidth=LINE_WIDTH,
                alpha=0.5,
                zorder=1,
            )

        if no_period < period_number - 1:
            ax.plot(
                traj[no_period + 1][:5, 4 * robot_id],
                traj[no_period + 1][:5, 4 * robot_id + 1],
                color=COLORS[robot_id],
                linestyle=LINE_STYLES[2],
                linewidth=LINE_WIDTH,
                alpha=0.5,
                zorder=1,
            )


def plot_trajs(axes, now_traj, infomap):
    now_distance = compute_distances(now_traj, dim=2, nx=dynamics_dim)
    now_traj = now_traj.reshape(period_number, -1, dynamics_dim * robot_number)
    for i in range(period_number):
        ax = axes[i]
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2.5)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2])
        # ax.set_title(f"({chr(97+i)}) Period {i+1}", pad=15)
        # else:
        #     ax.set_ylabel("Y Position (m)")
        # ax.set_xlabel("X Position (m)")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # 绘制障碍物
        # 绘制背景地图
        from matplotlib.colors import LinearSegmentedColormap

        # cmap = plt.get_cmap("Reds")
        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", ["white", "darkred"]
        )
        ax.contourf(
            infomap.domain[0],
            infomap.domain[1],
            infomap.evals[0].reshape(infomap.domain[0].shape),
            cmap=custom_cmap,
            levels=20,
            vmin=0.05,
            alpha=0.8,
        )
        # ax.contourf(
        #     infomap.domain[0],
        #     infomap.domain[1],
        #     infomap.evals[0].reshape(infomap.domain[0].shape),
        #     cmap="Greys",
        #     levels=20,
        #     # alpha=0.6,
        # )
        for obs_no in range(len(OBS_radius)):
            ax.add_patch(
                plt.Circle(
                    OBS_center[obs_no],
                    OBS_radius[obs_no],
                    color="darkred",
                    fill=False,
                    linewidth=1,
                    alpha=0.8,
                )
            )
            # 绘制被包裹的矩形
            ax.add_patch(
                plt.Rectangle(
                    OBS_center[obs_no] - np.array([OBS_rect[0] / 2, OBS_rect[1] / 2]),
                    OBS_rect[0],
                    OBS_rect[1],
                    color="lightblue",
                    fill=True,
                    alpha=0.6,
                )
            )

        plot_traj_period(now_traj, i, ax)

    ax_distance = axes[-1]
    ax_distance.set_xticklabels([])
    ax_distance.set_yticklabels([])

    # 创建时间序列
    time_steps = np.arange(0, len(now_distance[0, 0, :]) * 0.1, 0.1)

    for i, j in params["robot_pair"]:
        ax_distance.plot(time_steps, now_distance[i, j, :], linewidth=1.0)
    ax_distance.plot(
        time_steps,
        now_distance[0, 0] * 0 + params["com_radius"],
        "g--",
        label="$R_c$",
        linewidth=2.0,
    )
    ax_distance.plot(
        time_steps,
        now_distance[0, 0] * 0 + params["avoidance_radius"],
        "r--",
        label="$\\varepsilon_a$",
        linewidth=2.0,
    )
    ax_distance.tick_params(labelright=True, labelleft=False)
    ax_distance.set_ylim(0, now_distance.max() * 1.1)
    ax_distance.set_xlim(0, time_steps[-1] + 0.1)


# %%
