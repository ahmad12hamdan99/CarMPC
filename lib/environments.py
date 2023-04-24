import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np
from typing import Union

from lib.simulator import CarTrailerDimension


class BaseEnv:
    def __init__(self):
        self.constraints_A = []
        self.constraints_b = []
        self.goal = None
        self.lim = None
        self.name = 'BaseEnv'

        # Check the length of the constraints lists
        self.check_constraints()

        # For the legend of the plot
        goal_star = mlines.Line2D([], [], color='lime', marker='*', linestyle='None', markersize=10, label='goal')
        self.handles = [goal_star]

    def set_goal(self, goal: Union[list[float], np.ndarray, tuple[float]]):
        self.goal = goal

    def set_lim(self, lim: Union[tuple[tuple[int, int], tuple[int, int]], list[tuple[int, int]]]):
        """
        Set the ranges of the plot
        :param lim: the axes ranges of the plot [(x_min, x_max), (y_min, y_max)]
        """
        self.lim = lim

    def check_constraints(self):
        """
        This function check if the constraint matrices have the same length
        """
        assert len(self.constraints_A) == len(self.constraints_b), \
            f"The constraints array (len(A) = {len(self.constraints_A)} and len(B) = {len(self.constraints_b)})" + \
            "do not have the same length. This will produced unpredictable behaviour."

    def plot(self) -> None:
        """
        This function should add all the visualisation to the environment
        """
        self.x_lim, self.y_lim = self.lim
        plt.scatter(self.goal[0], self.goal[1], marker="*", c='lime')



class RoadEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        # -3 <= y <= 3
        self.constraints_A += [[0, 1, 0, 0], [0, -1, 0, 0]]
        self.constraints_b += [3, 3]
        self.y_lower, self.y_upper = -3, 3  # for plotting
        self.lim = [(-10, 50), (-10, 10)]
        self.name = 'RoadEnv'

        self.check_constraints()
        self.goal = [30, 1.5, 0, 0]

        # For the legend of the plot
        grey_patch = patches.Patch(color=(0.6, 0.6, 0.6), label='Road boundaries')
        self.handles += [grey_patch]

    def plot(self) -> None:
        """
        Calls the parents plot function and adds a visualisation for the road
        """
        super().plot()
        ax = plt.gca()
        # Road boundaries
        plt.fill_between([*self.x_lim], self.y_upper, self.y_lim[1], color=(0.6, 0.6, 0.6))
        plt.fill_between([*self.x_lim], self.y_lower, self.y_lim[0], color=(0.6, 0.6, 0.6))
        plt.plot(list(self.x_lim), [0, 0], color=(0, 0, 0), linestyle=(0, (5, 10)))


class RoadOneCarEnv(RoadEnv):
    def __init__(self):
        super().__init__()
        # Define constraints which involve multiple states: e.g. lines in the 2D plane
        # A @ x <= b
        self.constraints_A += [[1, 0, 0, 0]]
        self.constraints_b += [30]
        self.name = 'RoadOneCarEnv'

        self.check_constraints()
        self.goal = [29.9, -1.5, 0, 0]

        # For the legend of the plot
        yellow_patch = patches.Patch(color=(0.91, 0.8, 0.18), label='Obstacles')
        self.handles += [yellow_patch]

    def plot(self) -> None:
        """
        Calls the parents plot function and adds a visualisation for the other cars and belonging constraints
        """
        super().plot()
        ax = plt.gca()
        # Car 1
        x_upper = 30
        plt.vlines(x_upper, self.y_lim[0], self.y_lim[1], color=(0.91, 0.8, 0.18), linestyle='--')
        plt.fill_between([x_upper, self.x_lim[1]], self.y_lim[0], self.y_lim[1], color=(0.91, 0.8, 0.18, 0.1))
        for i in range(3):
            ax.add_patch(patches.Rectangle((
                                           30 + CarTrailerDimension.car_length - CarTrailerDimension.l12 + i * 1.25 * CarTrailerDimension.car_length,
                                           - 1.5 - CarTrailerDimension.car_width / 2),
                                           CarTrailerDimension.car_length,
                                           CarTrailerDimension.car_width,
                                           edgecolor='none',
                                           facecolor=(0.91, 0.8, 0.18),
                                           ))


class RoadMultipleCarsEnv(RoadEnv):
    def __init__(self):
        super().__init__()
        # Define constraints which involve multiple states: e.g. lines in the 2D plane
        # A @ x <= b
        self.constraints_A += [[-0.25, 1, 0, 0], [0.25, -1, 0, 0]]
        self.constraints_b += [-2, 6.25]
        self.name = 'RoadMultipleCarsEnv'

        self.check_constraints()
        self.goal = [30, 1.5, 0, 0]

        # For the legend of the plot
        yellow_patch = patches.Patch(color=(0.91, 0.8, 0.18), label='Obstacles')
        purple_patch = patches.Patch(color=(0.78, 0.18, 0.91), label='Obstacles')
        self.handles += [yellow_patch, purple_patch]

    def plot(self) -> None:
        """
        Calls the parents plot function and adds a visualisation for the other cars and belonging constraints
        """
        super().plot()
        ax = plt.gca()
        # Car 1
        car_constraint_A = self.constraints_A[-2]
        car_constraint_b = self.constraints_b[-2]
        x = np.linspace(*self.x_lim, 2)
        y = (-car_constraint_A[0] * x + car_constraint_b) / car_constraint_A[1]
        plt.plot(x, y, color=(0.91, 0.8, 0.18), linestyle='--')
        plt.fill_between(x, y, self.y_lim[1], color=(0.91, 0.8, 0.18, 0.1))
        for i in range(3):
            ax.add_patch(patches.Rectangle((-1 - i * 1.25 * CarTrailerDimension.car_length,
                                            1.5 - CarTrailerDimension.car_width / 2),
                                           CarTrailerDimension.car_length,
                                           CarTrailerDimension.car_width,
                                           edgecolor='none',
                                           facecolor=(0.91, 0.8, 0.18),
                                           ))

        # Car 2
        car_constraint_A = self.constraints_A[-1]
        car_constraint_b = self.constraints_b[-1]
        x = np.linspace(*self.x_lim, 2)
        y = (-car_constraint_A[0] * x + car_constraint_b) / car_constraint_A[1]
        plt.plot(x, y, color=(0.78, 0.18, 0.91), linestyle='--')
        plt.fill_between(x, y, self.y_lim[0], color=(0.78, 0.18, 0.91, 0.1))
        for i in range(4):
            ax.add_patch(patches.Rectangle((30 + i * 1.25 * CarTrailerDimension.car_length,
                                            -1.5 - CarTrailerDimension.car_width / 2),
                                           CarTrailerDimension.car_length,
                                           CarTrailerDimension.car_width,
                                           edgecolor='none',
                                           facecolor=(0.78, 0.18, 0.91),
                                           ))

