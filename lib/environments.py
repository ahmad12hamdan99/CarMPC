import matplotlib.pyplot as plt
import numpy as np
from typing import Union

# class BaseEnv:
#     def __init__(self,
#                  x_bounds: Union[list[float, float], tuple[float, float]],
#                  ):
#         self.x_lower, self.x_upper = x_bounds
#
#     def plot(self, ax) -> None:
#         """
#         This function should add all the visualisation to the environment
#         """
#         raise NotImplementedError
#
#     def constraints_mpc(self) -> list:
#         """
#         This function should return all the constraints for the MPC controller
#         :return:
#         """
#         raise NotImplementedError
#
#
# class FreeRoadEnv(BaseEnv):
#     def plot(self, ax) -> None:
#         x_lower, x_upper = -10, 30
#         y_lower, y_upper = -5, 5
#         plt.fill_between([*x_lim], [y_upper, y_upper], [y_lim[1], y_lim[1]], color=(0.5, 0.5, 0.5))
#         plt.fill_between([*x_lim], [y_lower, y_lower], [y_lim[0], y_lim[0]], color=(0.5, 0.5, 0.5))

#### Something like ax + by <= c
# y <= 5
# -y <= 5
