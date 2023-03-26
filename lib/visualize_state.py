import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from lib.model import CarTrailerDimension


def plot_graph(log_data: dict[str, np.ndarray], dt: float = 0.01) -> None:
    """
    Plots the states over a certain time range in a graph

    :param log_data: The log data in a dictionary
    :param dt: The timestep used to run the model
    :param car_dimension: The dimensions of the car
    :return: No return
    """
    time_range = np.arange(len(log_data['car'])) * dt
    # plt.figure()
    # plt.title(f"The position of the car plot vs. time (dt = {dt})")
    # plt.plot(time_range, log_data['car'][..., :2], label=['x', 'y'])
    # plt.legend()
    # plt.show()

    plt.figure()
    filter = 200
    x_car, y_car, psi_car = log_data['car'][::filter, 0], log_data['car'][::filter, 1], log_data['car'][::filter, 2]
    x_trailer, y_trailer, psi_trailer = log_data['trailer'][::filter, 0], log_data['trailer'][::filter, 1], log_data[
                                                                                                                'trailer'][
                                                                                                            ::filter, 2]
    ax = plt.gca()
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

    plt.title(f"The position of the car plot on a grid (dt = {dt})")
    plt.scatter(x_car, y_car, label='car', c='b')
    plt.scatter(x_trailer, y_trailer, label='trailer', c='r')
    plt.legend()
    plt.grid()
    ax.set_aspect('equal', 'box')
    plt.show()

    # plt.figure()
    # plt.title(f"The plot vs. time (dt = {dt})")
    # plt.plot(time_range, log_data['car'][..., 2:4], label=['psi', 'gamma'])
    # plt.legend()
    # plt.show()
