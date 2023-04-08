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
                 lim: Union[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int], tuple[int, int]]] = None,
                 env: BaseEnv = None
                 ) -> None:
    """
    Plots the states over a certain time range in a graph

    :param log_data: The log data in a dictionary
    :param dt: The timestep used to run the model
    :param goal: The goal state.
    :param lim: the axes ranges of the plot [(x_min, x_max), (y_min, y_max)]
    :param env: The environment that defines the road boundaries and obstacles
    :return: No return
    """
    lim = [(-40, 40), (-40, 40)] if lim is None else lim
    x_lim, y_lim = lim
    time_range = np.arange(len(log_data['car'])) * dt

    plt.figure()
    ax = plt.gca()
    if env is not None:
        env.plot(lim)

    interval = len(time_range) - 1
    x_car, y_car, psi_car = log_data['car'][::interval, 0], log_data['car'][::interval, 1], log_data['car'][::interval, 2]
    x_trailer, y_trailer, psi_trailer = log_data['trailer'][::interval, 0], log_data['trailer'][::interval, 1], \
        log_data['trailer'][::interval, 2]

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

        t_trailer = transforms.Affine2D().rotate_around(x_trailer[i], y_trailer[i], psi_trailer[i])
        c_trailer = ax.add_patch(
            patches.Rectangle((x_trailer[i] - (
                    CarTrailerDimension.trailer_length - CarTrailerDimension.l2),
                               y_trailer[i] - CarTrailerDimension.trailer_width / 2),
                              CarTrailerDimension.trailer_length - CarTrailerDimension.triangle_length,
                              CarTrailerDimension.trailer_width,
                              edgecolor='r',
                              facecolor='none',
                              ))
        c_trailer.set_transform(t_trailer + ax.transData)
        c_trailer_triangle = ax.add_patch(
            patches.Polygon([[x_trailer[i] + CarTrailerDimension.l2 - CarTrailerDimension.triangle_length,
                              y_trailer[i] - CarTrailerDimension.trailer_width / 2],
                             [x_trailer[i] + CarTrailerDimension.l2 - CarTrailerDimension.triangle_length,
                              y_trailer[i] + CarTrailerDimension.trailer_width / 2],
                             [x_trailer[i] + CarTrailerDimension.l2, y_trailer[i]]],
                            edgecolor='r',
                            facecolor='none',
                            ))
        c_trailer_triangle.set_transform(t_trailer + ax.transData)

    if goal is not None:
        plt.scatter(goal[0], goal[1], marker="*", c='lime')

    plt.title(f"The position of the car and trailer")
    plt.scatter(x_car, y_car, label='car', c='b')
    plt.scatter(x_trailer, y_trailer, label='trailer', c='r')
    plt.plot(log_data['car'][..., 0], log_data['car'][..., 1], label='car traject', c='b', linestyle='--')
    plt.plot(log_data['trailer'][..., 0], log_data['trailer'][..., 1], label='trailer traject', c='r', linestyle='--')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc='upper center', handles=handles + env.handles,
              bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=5)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    ax.set_aspect('equal', 'box')
    plt.show()


def plot_live(log_data: dict[str, np.ndarray],
              dt: float = 0.01,
              time=None,
              goal: np.ndarray = None,
              state_horizon: np.ndarray = None,
              lim: Union[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int], tuple[int, int]]] = None,
              env: BaseEnv = None,
              ) -> None:
    """
    Plots the states live in a refreshing graph

    :param log_data: The log data in a dictionary
    :param dt: The timestep used to run the model
    :param time: The current simulation time
    :param goal: The goal state.
    :param state_horizon: The planned horizon for state
    :param lim: the axes ranges of the plot [(x_min, x_max), (y_min, y_max)]
    :param env: The environment that defines the road boundaries and obstacles

    :return: No return
    """
    lim = [(-40, 40), (-40, 40)] if lim is None else lim
    x_lim, y_lim = lim

    plt.cla()
    ax = plt.gca()
    if env is not None:
        env.plot(lim)

    x_car, y_car, psi_car = log_data['car'][0], log_data['car'][1], log_data['car'][2]
    x_trailer, y_trailer, psi_trailer = log_data['trailer'][0], log_data['trailer'][1], log_data['trailer'][2]
    t_car = transforms.Affine2D().rotate_around(x_car, y_car, psi_car)
    c_car = ax.add_patch(
        patches.Rectangle((x_car - CarTrailerDimension.l12, y_car - CarTrailerDimension.car_width / 2),
                          CarTrailerDimension.car_length,
                          CarTrailerDimension.car_width,
                          edgecolor='b',
                          facecolor='none',
                          ))
    c_car.set_transform(t_car + ax.transData)

    t_trailer = transforms.Affine2D().rotate_around(x_trailer, y_trailer, psi_trailer)
    c_trailer = ax.add_patch(
        patches.Rectangle((x_trailer - (
                CarTrailerDimension.trailer_length - CarTrailerDimension.l2),
                           y_trailer - CarTrailerDimension.trailer_width / 2),
                          CarTrailerDimension.trailer_length - CarTrailerDimension.triangle_length,
                          CarTrailerDimension.trailer_width,
                          edgecolor='r',
                          facecolor='none',
                          ))
    c_trailer.set_transform(t_trailer + ax.transData)
    c_trailer_triangle = ax.add_patch(
        patches.Polygon([[x_trailer + CarTrailerDimension.l2 - CarTrailerDimension.triangle_length,
                          y_trailer - CarTrailerDimension.trailer_width / 2],
                         [x_trailer + CarTrailerDimension.l2 - CarTrailerDimension.triangle_length,
                          y_trailer + CarTrailerDimension.trailer_width / 2],
                         [x_trailer + CarTrailerDimension.l2, y_trailer]],
                        edgecolor='r',
                        facecolor='none',
                        ))
    c_trailer_triangle.set_transform(t_trailer + ax.transData)

    if goal is not None:
        plt.scatter(goal[0], goal[1], marker="*", c='lime')

    if state_horizon is not None:
        plt.scatter(state_horizon[:, 0], state_horizon[:, 1], marker="x", c='black')

    plt.title(f"The position of the car and trailer" if time is None
              else f"The position of the car and trailer (t = {time: .1f} s)")
    plt.scatter(x_car, y_car, label='car', c='b')
    plt.scatter(x_trailer, y_trailer, label='trailer', c='r')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(loc='upper center', handles=handles + env.handles,
              bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=5)
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    ax.set_aspect('equal', 'box')
    plt.pause(dt)


# def plot_environment(ax,
#                      x_lim: Union[tuple[int, int], list[int, int]],
#                      y_lim: Union[tuple[int, int], list[int, int]],
#                      ) -> None:
#     """
#     Plot the obstacles, the constraints and the road with its boundaries
#     :param ax: matplotlib axis object (return of plt.gca())
#     :param x_lim: The x range of the plot
#     :param y_lim: The y range of the plot
#     """
#     # Car 1
#     car_constraint_A = [-0.15, 1, 0, 0, 0]
#     car_constraint_b = -1.5
#     x = np.linspace(*x_lim, 2)
#     y = (-car_constraint_A[0] * x + car_constraint_b) / car_constraint_A[1]
#     plt.plot(x, y, color=(0.91, 0.8, 0.18), linestyle='--')
#     plt.fill_between(x, y, y_lim[1], color=(0.91, 0.8, 0.18, 0.1))
#     for i in range(3):
#         ax.add_patch(patches.Rectangle((0 - i * 1.25 * CarTrailerDimension.car_length,
#                                         1.25 - CarTrailerDimension.car_width / 2),
#                                        CarTrailerDimension.car_length,
#                                        CarTrailerDimension.car_width,
#                                        edgecolor='none',
#                                        facecolor=(0.91, 0.8, 0.18),
#                                        ))
#
#     # Car 2
#     car_constraint_A = [-0.15, 1, 0, 0, 0]
#     car_constraint_b = -3.75
#     x = np.linspace(*x_lim, 2)
#     y = (-car_constraint_A[0] * x + car_constraint_b) / car_constraint_A[1]
#     plt.plot(x, y, color=(0.78, 0.18, 0.91), linestyle='--')
#     plt.fill_between(x, y, y_lim[0], color=(0.78, 0.18, 0.91, 0.1))
#     for i in range(4):
#         ax.add_patch(patches.Rectangle((30 + i * 1.25 * CarTrailerDimension.car_length,
#                                         -1.25 - CarTrailerDimension.car_width / 2),
#                                        CarTrailerDimension.car_length,
#                                        CarTrailerDimension.car_width,
#                                        edgecolor='none',
#                                        facecolor=(0.78, 0.18, 0.91),
#                                        ))
#
#     # Road boundaries
#     y_lower, y_upper = -3, 3
#     plt.fill_between([*x_lim], y_upper, y_lim[1], color=(0.6, 0.6, 0.6))
#     plt.fill_between([*x_lim], y_lower, y_lim[0], color=(0.6, 0.6, 0.6))
#     plt.plot(list(x_lim), [0, 0], color=(0, 0, 0), linestyle=(0, (5, 10)))


def plot_graphs(log_data: dict[str, np.ndarray], dt: float = 0.01) -> None:
    """
    Plots graphs of the different states and inputs against the time

    :param log_data: The log data in a dictionary
    :param dt: The timestep used to run the model
    :return: No return
    """
    plt.figure()
    time_range = np.arange(len(log_data['car'])) * dt
    fig, axs = plt.subplots(5, 1, figsize=(5, 15), constrained_layout=True)
    fig.suptitle("Car states")
    titles = ['x', 'y', '\psi', '\gamma', 'v']
    units = ['m', 'm', 'rad', 'rad', 'm/s']
    for i in range(len(axs)):
        ax = axs[i]
        ax.plot(time_range, log_data['car'][..., i])
        ax.set_title(f'${titles[i]}$')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f'${titles[i]}$' + ' [' + units[i] + ']')

    plt.show()
