import numpy as np
import cvxpy as cp
import math
from typing import Union

from lib.simulator import CarTrailerDimension
from lib.matrix_gen import predmod, costgen


class MPC:
    def __init__(self,
                 dt: float,
                 N: int,
                 lin_state: Union[np.ndarray, list],
                 lin_input: Union[np.ndarray, list],
                 ) -> None:
        """
        Initialize the MPC controller
        :param dt: The timestep used to discretize the state space model
        :param N: The prediction horizon length used in the MPC controller
        :param lin_state: The state around which to linearize the state space model
        :param lin_input: The input around which to linearize the state space model
        """
        self.dt = dt
        self.N = N
        self.nx, self.nu = 5, 2
        self.Q = np.diag([3, 3, 1, 1, 1])
        self.R = np.diag([2, 0])
        self.P = np.zeros((5, 5))  # Should be result from DARE equation
        self.goal = np.array([0, 0, 0, 0, 0])
        self.x_horizon = None

        self.lin_state = np.array(lin_state) if isinstance(lin_state, list) else lin_state
        self.lin_input = np.array(lin_input) if isinstance(lin_input, list) else lin_input
        # Init the model
        A_lin, B_lin = self.linearized_model(self.lin_state, self.lin_input)
        self.A, self.B = self.discretized_model(A_lin, B_lin, dt)

        # Generate the prediction model and cost function matrices
        self.T, self.S = predmod(self.A, self.B, self.N)
        self.H, self.h, _ = costgen(self.Q, self.R, self.P, self.T, self.S, self.nx)

    @staticmethod
    def linearized_model(lin_state: np.ndarray, lin_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        This function returns the linearized state space model for the car with a trailer dynamics.
        The original model is from (equation 1): https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=281527b8c72787dd51f167d0669cb31097bd21f6
        I added the velocity to make it a bit broader usable.
        :param lin_state: The state around which to linearize
        :param lin_input: The input around which to linearize
        :return: The A and B matrix (in x' = Ax + Bu) of the linearized model
        """
        assert lin_state.shape == (5,), f"The state should have shape [5], but has {lin_state.shape}."
        assert lin_input.shape == (2,), f"The input should have shape [2], but has {lin_input.shape}."

        x_e, y_e, psi_e, gamma_e, v_e = lin_state
        a_e, delta_f_e = lin_input
        l1, l2, l12 = CarTrailerDimension.l1, CarTrailerDimension.l2, CarTrailerDimension.l12
        A_lin = np.array([[0, 0, -v_e * np.sin(psi_e), 0, np.cos(psi_e)],
                          [0, 0, v_e * np.cos(psi_e), 0, np.sin(psi_e)],
                          [0, 0, 0, 0, np.tan(delta_f_e) / l1],
                          [0, 0, 0,
                           -v_e * l12 / (l1 * l2) * np.sin(gamma_e) * np.tan(delta_f_e) - v_e / l2 * np.sin(gamma_e),
                           (1 / l1 + l12 / (l1 * l1) * np.cos(gamma_e)) * np.tan(delta_f_e) - np.sin(gamma_e) / l2],
                          [0, 0, 0, 0, 0]])
        B_lin = np.array([[0, 0],
                          [0, 0],
                          [0, v_e / l1 * 1 / np.cos(delta_f_e) ** 2],
                          [0, (v_e / l1 + v_e * l12 / (l1 * l2) * np.cos(gamma_e)) / np.cos(delta_f_e) ** 2],
                          [1, 0]])

        return A_lin, B_lin

    @staticmethod
    def discretized_model(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the discretized state space model of a continues time state space model. It uses the Euler (forward) method:
        x+ = x + dt * x' = x + dt * (Ax + Bu) = (dt * A + I)x + dt * Bu = A_d x + B_d u
        Which gives:
            A_d = (dt * A + I)
            B_d = dt * B
        :param A: The A matrix (in x' = Ax + Bu)
        :param B: The B matrix (in x' = Ax + Bu)
        :param dt: The timestep to use when linearizing
        :return: Returns the A and B matrices (in x+ = Ax + Bu)
        """
        assert len(A) == len(B), "Matrices A and B should have the same height."
        A_d = dt * A + np.eye(len(A))
        B_d = dt * B

        return A_d, B_d

    def set_goal(self, goal: Union[np.ndarray, list]) -> None:
        """
        Set the goal of the MPC controller
        :param goal: the goal state. Should be an equilibrium state
        :return:
        """
        assert np.array(goal).shape == (5,)
        # check if the goal is an equilibrium state: x+ = x with u = 0
        assert np.all(goal == self.A @ goal)
        # Since it is linearized around a speed > 0, this does not really work, bc psi != throws an error

        self.goal = np.array(goal) if isinstance(goal, list) else goal

    def constraints(self):
        return []

    def step(self, x0) -> np.ndarray:
        """
        Function to step the mpc controller
        :param x0: the current state
        :return: returns the first action of the optimal control sequence
        """
        u = cp.Variable((self.N * self.nu))
        x0 = x0 - self.goal
        objective = cp.Minimize(1 / 2 * cp.quad_form(u, self.H) + self.h @ x0 @ u)
        constraints = self.constraints()
        problem = cp.Problem(objective, constraints)
        result = problem.solve()
        if math.isinf(result):
            assert False, "No feasible solution"

        self.x_horizon = self.T @ x0 + self.S @ u.value
        self.x_horizon = self.x_horizon.reshape(-1, self.nx) + self.goal

        return u.value[:2]
