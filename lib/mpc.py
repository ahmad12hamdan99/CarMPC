import numpy as np
import cvxpy as cp
import math
from typing import Union

import numpy.linalg
from scipy.linalg import solve_discrete_are as dare

from lib.simulator import CarTrailerDimension
from lib.matrix_gen import predmod, costgen
from lib.environments import BaseEnv


class MPC:
    def __init__(self,
                 dt: float,
                 N: int,
                 lin_state: Union[np.ndarray, list],
                 lin_input: Union[np.ndarray, list],
                 env: BaseEnv = None,
                 ) -> None:
        """
        Base class for an MPC controller
        :param dt: The timestep used to discretize the state space model
        :param N: The prediction horizon length used in the MPC controller
        :param lin_state: The state around which to linearize the state space model
        :param lin_input: The input around which to linearize the state space model
        :param env: The environment which also holds constraints
        """
        self.dt = dt
        self.N = N
        self.lin_state = np.array(lin_state) if isinstance(lin_state, list) else lin_state
        self.lin_input = np.array(lin_input) if isinstance(lin_input, list) else lin_input
        self.goal = np.array([0, 0, 0, 0, 0])
        self.x_horizon, self.u_horizon = None, None
        self.env = env

        # Linearize and discretize the model
        A_lin, B_lin = self.linearized_model(self.lin_state, self.lin_input)
        self.A, self.B = self.discretized_model(A_lin, B_lin, dt)

        # Init state constraint matrices
        self.state_const_A, self.state_const_b = [], []

    def check_constraints(self):
        """
        This function check if the constraint matrices have the same length
        """
        assert len(self.state_const_A) == len(self.state_const_b), \
            f"The constraints array (len(A) = {len(self.state_const_A)} and len(B) = {len(self.state_const_b)})" + \
            "do not have the same length. This will produced unpredictable behaviour."

    @staticmethod
    def linearized_model(lin_state: np.ndarray, lin_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        This function returns the linearized state space model for the car with a trailer dynamics.
        The original model is from (equation 1): https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=281527b8c72787dd51f167d0669cb31097bd21f6
        The velocity was added to make it a bit broader usable.
            state x: [x, y, psi, gamma, v]
                x: x-position of the car
                y: y-position of the car
                psi: orientation of the car in the 2D plane
                v: velocity of the car
            inputs u: [a, delta_f]
                a: acceleration of the car
                delta_f: steering angle of the car

        :param lin_state: The state around which to linearize
        :param lin_input: The input around which to linearize
        :return: The A and B matrix (in x' = Ax + Bu) of the linearized model
        """
        assert lin_state.shape == (4,), f"The state should have shape [4], but has {lin_state.shape}."
        assert lin_input.shape == (2,), f"The input should have shape [2], but has {lin_input.shape}."

        x_e, y_e, psi_e, v_e = lin_state
        a_e, delta_f_e = lin_input
        l1, l2, l12 = CarTrailerDimension.l1, CarTrailerDimension.l2, CarTrailerDimension.l12
        A_lin = np.array([[0, 0, -v_e * np.sin(psi_e), np.cos(psi_e)],
                          [0, 0, v_e * np.cos(psi_e), np.sin(psi_e)],
                          [0, 0, 0, np.tan(delta_f_e) / l1],
                          [0, 0, 0, 0]])
        B_lin = np.array([[0, 0],
                          [0, 0],
                          [0, v_e / l1 * 1 / np.cos(delta_f_e) ** 2],
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
        assert np.array(goal).shape == (4,)
        # check if the goal is an equilibrium state: x+ = x with u = 0
        assert np.all(goal == self.A @ goal), \
            "Goal is not an equilibrium point. At least not for the linearized model. E.g. psi != 0 is not an equilibrium point."
        # Since it is linearized around a speed > 0, this is quite strict, because psi != 0 throws an error

        self.goal = np.array(goal) if isinstance(goal, list) else goal


class MPCStateFB(MPC):
    def __init__(self,
                 dt: float,
                 N: int,
                 lin_state: Union[np.ndarray, list],
                 lin_input: Union[np.ndarray, list],
                 terminal_constraint: bool = True,
                 input_constraint: bool = True,
                 state_constraint: bool = True,
                 env: BaseEnv = None,
                 ) -> None:
        """
        Initialize the MPC controller with state feedback
        :param dt: The timestep used to discretize the state space model
        :param N: The prediction horizon length used in the MPC controller
        :param lin_state: The state around which to linearize the state space model
        :param lin_input: The input around which to linearize the state space model
        :param env: The environment which also holds constraints

        """
        super().__init__(dt, N, lin_state, lin_input, env)

        self.nx, self.nu = 4, 2
        self.Q = np.diag([3, 3, 10, 11])  # [1, 1, 10, 10, 3] # [3, 3, 1, 1, 1]
        self.R = np.eye(self.nu) * 10

        self.terminal_constraint_bool = terminal_constraint
        self.input_constraint_bool = input_constraint
        self.state_constraint_bool = state_constraint

        # Solve the discrete algebraic Riccati equation
        try:
            self.P = dare(self.A, self.B, self.Q, self.R)
        except numpy.linalg.LinAlgError:
            # print("\nNOT POSSIBLE TO SOLVE DARE\n")
            assert False, ("\nNOT POSSIBLE TO SOLVE DARE\n")
            self.P = np.zeros((4, 4))
        # Generate the prediction model and cost function matrices
        self.T, self.S = predmod(self.A, self.B, self.N)
        self.H, self.h, _ = costgen(self.Q, self.R, self.P, self.T, self.S, self.nx)

        # Define input constraints: [acceleration, steering angle]
        self.input_upper = np.array([2, np.pi / 8])
        self.input_lower = -self.input_upper

        # Define state constraints:
        # self.x_lower, self.x_upper = -30, 50
        # self.y_lower, self.y_upper = -3, 3
        # self.v_lower, self.v_upper = 0, 5
        # self.psi_lower, self.psi_upper = -np.pi/4, np.pi/4

        # State constraints to stay close to the linearized state:
        # -pi/4 <= psi <= pi/4
        self.state_const_A += [[0, 0, 1, 0], [0, 0, -1, 0]]
        self.state_const_b += [np.pi / 8, np.pi / 8]
        # 0 <= v <= 5
        self.state_const_A += [[0, 0, 0, 1], [0, 0, 0, -1]]
        self.state_const_b += [5, 0]


        # Check lengths of constraints
        self.check_constraints()

    def terminal_constraint(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the constraint for the terminal set in this function
        A x <= b
            x = the state sequence with a variable length depending on the horizon
        :param x: the state sequence; the type is some cvxpy expression
        :param u: the input sequence; the type is some cvxpy expression
        :return: cvxpy compatible constraint
        """
        # Add the terminal constraint
        A = np.zeros((2 * self.nx, (self.N + 1) * self.nx))
        # A[0:self.nx, self.N * self.nx:(self.N + 1) * self.nx] = np.eye(self.nx)  # state <= ...
        # A[self.nx: 2 * self.nx, self.N * self.nx:(self.N + 1) * self.nx] = -np.eye(self.nx)  # -state <= ...

        # ### ONLY CONSTRAINT ON X AND Y
        # A[0:self.nx, self.N * self.nx:(self.N + 1) * self.nx] = np.diag([1, 1, 0, 0])  # state <= ...
        # A[self.nx: 2 * self.nx, self.N * self.nx:(self.N + 1) * self.nx] = -np.diag([1, 1, 0, 0])  # -state <= ...

        # ### ONLY CONSTRAINT ON X, Y AND PSI
        A[0:self.nx, self.N * self.nx:(self.N + 1) * self.nx] = np.diag([1, 1, 1, 0])  # state <= ...
        A[self.nx: 2 * self.nx, self.N * self.nx:(self.N + 1) * self.nx] = -np.diag([1, 1, 1, 0])  # -state <= ...

        b = np.hstack((self.goal, -self.goal))

        A, b = np.vstack(A).squeeze(), np.vstack(b).squeeze()
        return A, b

    def input_constraint(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the constraint for inputs in this function
        A u <= b
            u = the input sequence with a variable length depending on the horizon
        :return: cvxpy compatible constraint
        """
        assert self.nu == len(self.input_upper), \
            f"Number of inputs ({self.nu}) is different from number of upper inputs boundaries ({len(self.input_upper)})"
        assert self.nu == len(self.input_lower), \
            f"Number of inputs ({self.nu}) is different from number of upper inputs boundaries ({len(self.input_lower)})"

        A = np.vstack((np.eye(self.N * self.nu), -np.eye(self.N * self.nu)))
        b = np.vstack(((self.input_upper,) * self.N, (-self.input_lower,) * self.N)).flatten()

        return A, b

    def state_constraint(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the constraint for state (except from the terminal constraint) in this function
        A x <= b
            x = the state sequence with a variable length depending on the horizon
        :return: cvxpy compatible constraint
        """
        A, b = [], []

        for A_con, b_con in zip(self.state_const_A, self.state_const_b):
            A_hat = np.zeros((self.N, (self.N + 1) * self.nx))
            for i in range(self.N):
                A_hat[i, (i + 1) * self.nx:(i + 2) * self.nx] = A_con
            A.append(A_hat)

            b_hat = np.hstack(((b_con,) * self.N))
            b.append(b_hat)

        if self.env is not None:
            for A_con, b_con in zip(self.env.constraints_A, self.env.constraints_b):
                A_hat = np.zeros((self.N, (self.N + 1) * self.nx))
                for i in range(self.N):
                    A_hat[i, (i + 1) * self.nx:(i + 2) * self.nx] = A_con
                A.append(A_hat)

                b_hat = np.hstack(((b_con,) * self.N))
                b.append(b_hat)
        # Constraint on gamma to prevent jackknifing of the car and trailer
        # on all states, also x(0): does not really make sense since it cannot be steered, but shouldn't be a problem anyway
        # ======= DOES NOT WORK WITH THE LINEARIZED MODEL =======
        # A_gamma = np.zeros((2 * (self.N + 1), (self.N + 1) * self.nx))
        # di_1 = (np.arange(self.N + 1), np.arange(self.N + 1) * self.nx + 3)
        # A_gamma[di_1] = 1
        # di_2 = (np.arange(self.N + 1) + (self.N + 1), np.arange(self.N + 1) * self.nx + 3)
        # A_gamma[di_2] = -1
        # A.append(A_gamma)
        #
        # b_gamma = np.hstack(((self.gamma_upper,) * (self.N + 1), (-self.gamma_lower,) * (self.N + 1)))
        # b.append(b_gamma)

        # ============== TODO: SHOULD REMOVE ALL CONSTRAINTS ON x(0)
        # x upper and lower bounds
        # A_x = np.zeros((2 * (self.N + 1), (self.N + 1) * self.nx))
        # di_1 = (np.arange(self.N + 1), np.arange(self.N + 1) * self.nx + 0)
        # A_x[di_1] = 1
        # di_2 = (np.arange(self.N + 1) + (self.N + 1), np.arange(self.N + 1) * self.nx + 0)
        # A_x[di_2] = -1
        # A.append(A_x)
        #
        # b_x = np.hstack(((self.x_upper,) * (self.N + 1), (-self.x_lower,) * (self.N + 1)))
        # b.append(b_x)

        # y upper and lower bounds
        # A_y = np.zeros((2 * (self.N + 1), (self.N + 1) * self.nx))
        # di_1 = (np.arange(self.N + 1), np.arange(self.N + 1) * self.nx + 1)
        # A_y[di_1] = 1
        # di_2 = (np.arange(self.N + 1) + (self.N + 1), np.arange(self.N + 1) * self.nx + 1)
        # A_y[di_2] = -1
        # A.append(A_y)
        #
        # b_y = np.hstack(((self.y_upper,) * (self.N + 1), (-self.y_lower,) * (self.N + 1)))
        # b.append(b_y)

        # v upper and lower bounds
        # A_v = np.zeros((2 * (self.N + 1), (self.N + 1) * self.nx))
        # di_1 = (np.arange(self.N + 1), np.arange(self.N + 1) * self.nx + 4)
        # A_v[di_1] = 1
        # di_2 = (np.arange(self.N + 1) + (self.N + 1), np.arange(self.N + 1) * self.nx + 4)
        # A_v[di_2] = -1
        # A.append(A_v)
        #
        # b_v = np.hstack(((self.v_upper,) * (self.N + 1), (-self.v_lower,) * (self.N + 1)))
        # b.append(b_v)

        # psi upper and lower bounds
        # A_psi = np.zeros((2 * (self.N + 1), (self.N + 1) * self.nx))
        # di_1 = (np.arange(self.N + 1), np.arange(self.N + 1) * self.nx + 2)
        # A_psi[di_1] = 1
        # di_2 = (np.arange(self.N + 1) + (self.N + 1), np.arange(self.N + 1) * self.nx + 2)
        # A_psi[di_2] = -1
        # A.append(A_psi)
        #
        # b_psi = np.hstack(((self.psi_upper,) * (self.N + 1), (-self.psi_lower,) * (self.N + 1)))
        # b.append(b_psi)

        # additional constraint: -0.15x + y <= -1.5
        # car_constraint_A = [-0.15, 1, 0, 0, 0]
        # car_constraint_b = -1.5
        # A_car = np.zeros((self.N, (self.N + 1) * self.nx))
        # for i in range(self.N):
        #     A_car[i, (i + 1) * self.nx:(i + 2) * self.nx] = car_constraint_A
        # A.append(A_car)
        #
        # b_car = np.hstack(((car_constraint_b,) * (self.N)))
        # b.append(b_car)

        # additional constraint: -0.15x + y >= -3.75 --> 0.15x - y <= 3.75
        # car_constraint_A = [0.15, -1, 0, 0, 0]
        # car_constraint_b = 3.75
        # A_car = np.zeros((self.N, (self.N + 1) * self.nx))
        # for i in range(self.N):
        #     A_car[i, (i + 1) * self.nx:(i + 2) * self.nx] = car_constraint_A
        # A.append(A_car)
        #
        # b_car = np.hstack(((car_constraint_b,) * (self.N)))
        # b.append(b_car)

        A = np.vstack(A)
        b = np.hstack(b)

        return A, b

    def step(self, x0) -> np.ndarray:
        """
        Function to step the mpc controller
        :param x0: the current state
        :return: returns the first action of the optimal control sequence
        """

        u = cp.Variable((self.N * self.nu))
        x = self.T @ x0 + self.S @ u
        x0 = x0 - self.goal
        objective = cp.Minimize(1 / 2 * cp.quad_form(u, self.H) + self.h @ x0 @ u)

        constraints = []
        if self.terminal_constraint_bool:
            A, b = self.terminal_constraint()
            constraints += [A @ x <= b]
        if self.input_constraint_bool:
            A, b = self.input_constraint()
            constraints += [A @ u <= b]
        if self.state_constraint_bool:
            A, b = self.state_constraint()
            constraints += [A @ x <= b]

        problem = cp.Problem(objective, constraints)
        result = problem.solve()
        if math.isinf(result):
            # return [0, 0]
            assert False, "No feasible solution"

        # Save for visualisation purposes
        self.x_horizon = self.T @ x0 + self.S @ u.value
        self.x_horizon = self.x_horizon.reshape(-1, self.nx) + self.goal
        self.u_horizon = u.value.reshape(-1, 2)

        # if True:
        #     K = np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A
        #     return K @ x0

        return u.value[:self.nu]
