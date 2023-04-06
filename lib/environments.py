import matplotlib.pyplot as plt
import numpy as np
from typing import Union


class BaseEnv:
    def __init__(self):
        self.x_lower, self.x_upper = None, None
        self.y_lower, self.y_upper = None, None
        self.psi_lower, self.psi_upper = None, None
        self.gamma_lower, self.gamma_upper = None, None
        self.v_lower, self.v_upper = None, None
        self.x_lim, self.y_lim = (None, None), (None, None)

    def plot(self) -> None:
        """
        This function should add all the visualisation to the environment
        """
        raise NotImplementedError

    def constraints_mpc(self) -> list:
        """
        This function should return all the constraints for the MPC controller
        :return:
        """
        raise NotImplementedError


class FreeRoadEnv(BaseEnv):
    def __init__(self):
        super().__init__()
        self.y_lower, self.y_upper = -3, 3

    def plot(self) -> None:
        # Road boundaries
        plt.fill_between([*self.x_lim], self.y_upper, self.y_lim[1], color=(0.6, 0.6, 0.6))
        plt.fill_between([*self.x_lim], self.y_lower, self.y_lim[0], color=(0.6, 0.6, 0.6))
        plt.plot(list(self.x_lim), [0, 0], color=(0, 0, 0), linestyle=(0, (5, 10)))


class RoadWithObstaclesEnv(FreeRoadEnv):
    def __init__(self):
        super().__init__()
        # Define constraints which involve multiple states: e.g. lines in the 2D plane
        # A @ x <= b
        self.constraints_A = [[-0.15, 1, 0, 0, 0], [-0.15, 1, 0, 0, 0]]
        self.constraints_b = [-1.5, -3.75]

    def plot(self) -> None:
        super().plot()
        for A, b in zip(self.constraints_A, self.constraints_b):
            pass
