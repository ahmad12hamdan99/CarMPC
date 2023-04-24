import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from typing import Union

from lib.simulator import CarTrailerDimension
from lib.environments import BaseEnv


def plot_history(log_data: dict[str, np.ndarray],
                 dt: float = 0.01,
                 goal: np.ndarray = None,
                 env: BaseEnv = None
                 ) -> None:
    """
    Plots the states over a certain time range in a graph

    :param log_data: The log data in a dictionary
    :param dt: The timestep used to run the model
    :param goal: The goal state.
    :param env: The environment that defines the road boundaries and obstacles
    :return: No return
    """
    lim = [(-40, 40), (-40, 40)] if env.lim is None else env.lim
    x_lim, y_lim = lim
    time_range = np.arange(len(log_data['car'])) * dt

    plt.figure()
    ax = plt.gca()
    if env is not None:
        env.plot()

    interval = len(time_range) - 1
    x_car, y_car, psi_car = log_data['car'][::interval, 0], log_data['car'][::interval, 1], log_data['car'][::interval, 2]

    for i in range(len(x_car)):
        t_car = transforms.Affine2D().rotate_around(x_car[i], y_car[i], psi_car[i])
        c_car = ax.add_patch(
            patches.Rectangle((x_car[i] - CarTrailerDimension.l12, y_car[i] - CarTrailerDimension.car_width / 2),
                              CarTrailerDimension.car_length,
                              CarTrailerDimension.car_width,
                              edgecolor='b',
                              facecolor='none',
                              ))
        c_car.set_transform(t_car + ax.transData)

    if goal is not None:
        plt.scatter(goal[0], goal[1], marker="*", c='lime')

    plt.title(f"The position of the car and trailer")
    plt.scatter(x_car, y_car, label='car', c='b')
    plt.plot(log_data['car'][..., 0], log_data['car'][..., 1], label='true traject', c='b', linestyle='--')
    if 'estimate' in log_data.keys():
        plt.plot(log_data['estimate'][..., 0], log_data['estimate'][..., 1], label='est. traject', c=(1, 0, 0, 0.5), linestyle='--')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc='upper center', handles=handles + env.handles,
              bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=4)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    ax.set_aspect('equal', 'box')
    plt.show()


def plot_live(log_data: dict[str, np.ndarray],
              dt: float = 0.01,
              time=None,
              estimate: np.ndarray = None,
              state_horizon: np.ndarray = None,
              env: BaseEnv = None,
              ) -> None:
    """
    Plots the states live in a refreshing graph

    :param log_data: The log data in a dictionary
    :param dt: The timestep used to run the model
    :param time: The current simulation time
    :param estimate: the estimate state by the MPC controller.
    :param state_horizon: The planned horizon for state
    :param env: The environment that defines the road boundaries and obstacles

    :return: No return
    """
    lim = [(-40, 40), (-40, 40)] if env.lim is None else env.lim
    x_lim, y_lim = lim

    plt.cla()
    ax = plt.gca()
    if env is not None:
        env.plot()

    x_car, y_car, psi_car = log_data['car'][0], log_data['car'][1], log_data['car'][2]
    t_car = transforms.Affine2D().rotate_around(x_car, y_car, psi_car)
    c_car = ax.add_patch(
        patches.Rectangle((x_car - CarTrailerDimension.l12, y_car - CarTrailerDimension.car_width / 2),
                          CarTrailerDimension.car_length,
                          CarTrailerDimension.car_width,
                          edgecolor='b',
                          facecolor='none',
                          ))
    c_car.set_transform(t_car + ax.transData)

    if estimate is not None:
        t_car = transforms.Affine2D().rotate_around(*estimate[:3])
        c_car = ax.add_patch(
            patches.Rectangle((x_car - CarTrailerDimension.l12, y_car - CarTrailerDimension.car_width / 2),
                              CarTrailerDimension.car_length,
                              CarTrailerDimension.car_width,
                              edgecolor=(1, 0, 0, 0.5),
                              facecolor='none',
                              ))
        c_car.set_transform(t_car + ax.transData)
        plt.scatter(x_car, y_car, label='state est.', c=[(1, 0, 0, 0.5)])

    if state_horizon is not None:
        plt.scatter(state_horizon[:, 0], state_horizon[:, 1], marker="x", c='black', label='pred. horz.')

    plt.title(f"The position of the car and trailer" if time is None
              else f"The position of the car and trailer (t = {time: .1f} s)")
    plt.scatter(x_car, y_car, label='car', c='b')
    handles, _ = ax.get_legend_handles_labels()

    ax.legend(loc='upper center', handles=handles + env.handles,
              bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=4)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    ax.set_aspect('equal', 'box')
    plt.pause(dt)


def plot_graphs(log_data: dict[str, np.ndarray], dt: float = 0.01) -> None:
    """
    Plots graphs of the different states and inputs and the cost against the time

    :param log_data: The log data in a dictionary
    :param dt: The timestep used to run the model
    :return: No return
    """
    time_range = np.arange(len(log_data['car'])) * dt
    fig, axs = plt.subplots(4, 1, figsize=(5, 12), constrained_layout=True)
    fig.suptitle("Car states")
    titles = ['x', 'y', '\psi', 'v']
    units = ['m', 'm', 'rad', 'm/s']
    for i in range(4):
        ax = axs[i]
        ax.step(time_range, log_data['car'][..., i], label='true state')
        if 'estimate' in log_data.keys():
            ax.step(time_range, log_data['estimate'][..., i], label='state estimate')
        ax.set_title(f'${titles[i]}$')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f'${titles[i]}$' + ' [' + units[i] + ']')
        ax.legend()

    plt.show()
