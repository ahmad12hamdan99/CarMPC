import os.path

import numpy as np
import cvxpy as cp
import math
from typing import Union

import numpy.linalg
from scipy.linalg import solve_discrete_are as dare

from lib.simulator import CarTrailerDimension
from lib.matrix_gen import predmod, costgen
from lib.environments import BaseEnv
from lib.configuration import *


class OutsideTheRegionOfAttractionError(Exception):
    """The solver cannot find a solution so the init state lies outside the region of attraction (calligraph X}"""


class MPC:
    def __init__(self,
                 dt: float,
                 N: int,
                 lin_state: Union[np.ndarray, list],
                 lin_input: Union[np.ndarray, list],
                 terminal_constraint: bool = True,
                 input_constraint: bool = True,
                 state_constraint: bool = True,
                 env: BaseEnv = None,
                 use_LQR: bool = False,
                 ) -> None:
        """
        Base class for an MPC controller
        :param dt: The timestep used to discretize the state space model
        :param N: The prediction horizon length used in the MPC controller
        :param lin_state: The state around which to linearize the state space model
        :param lin_input: The input around which to linearize the state space model
        :param env: The environment which also holds constraints
        :param use_LQR: if true, an unconstrained LQR controller is used
        """
        self.dt = dt
        self.N = N
        self.lin_state = np.array(lin_state) if isinstance(lin_state, list) else lin_state
        self.lin_input = np.array(lin_input) if isinstance(lin_input, list) else lin_input
        self.goal = np.array(env.goal)
        self.x_horizon, self.u_horizon = None, None
        self.cost = 0
        self.env = env
        self.use_LQR = use_LQR

        # Linearize and discretize the model
        A_lin, B_lin = self.linearized_model(self.lin_state, self.lin_input)
        self.A, self.B = self.discretized_model(A_lin, B_lin, dt)

        # Init state constraint matrices
        self.state_const_A, self.state_const_b = [], []

        self.nx, self.nu = 4, 2
        self.Q = STAGE_COST_Q
        self.R = STAGE_COST_R

        self.terminal_constraint_bool = terminal_constraint
        self.input_constraint_bool = input_constraint
        self.state_constraint_bool = state_constraint

        # Solve the discrete algebraic Riccati equation

        try:
            self.P = dare(self.A, self.B, self.Q, self.R)
        except numpy.linalg.LinAlgError:
            print("\nNOT POSSIBLE TO SOLVE DARE\n")
            self.P = np.zeros((4, 4))
            # assert False, ("\nNOT POSSIBLE TO SOLVE DARE\n")

        # Generate the prediction model and cost function matrices
        self.T, self.S = predmod(self.A, self.B, self.N)
        self.H, self.h, _ = costgen(self.Q, self.R, self.P, self.T, self.S, self.nx)

        # Optimal unconstrained control law: u = K @ x0
        self.K = -np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A

        # Define input constraints: [acceleration, steering angle]
        self.input_upper = np.array([2, np.pi / 8])
        self.input_lower = -self.input_upper

        # State constraints to stay close to the linearized state:
        # -pi/8 <= psi <= pi/8
        self.state_const_A += [[0, 0, 1, 0], [0, 0, -1, 0]]
        self.state_const_b += [np.pi / 8, np.pi / 8]
        # -1 <= v <= 5
        self.state_const_A += [[0, 0, 0, 1], [0, 0, 0, -1]]
        self.state_const_b += [5, 1]

        # The terminal set constraints
        if terminal_constraint:
            filename = env.name + ''.join('_' + str(s) for s in env.goal) + '.npy'
            directory = '../terminal_sets/'
            try:
                file = os.path.join(directory, filename)
                terminal_set = np.load(file)
                A, b = terminal_set[..., :4], terminal_set[..., 4]
                A_hat = np.zeros((len(terminal_set), (self.N + 1) * self.nx))
                A_hat[:, -4:] = A
                self.term_set_A, self.term_set_b = A_hat, b
                print(f"Using {file} for the terminal set.")
            except FileNotFoundError:
                print(
                    f"No terminal set found for the specific combination of environment{env.name} and goal{env.goal}.",
                    f"The goal state will be used as terminal constraint, but this might be too strict.")
                # ### ONLY CONSTRAINT ON X, Y AND PSI
                A = np.zeros((2 * self.nx, (self.N + 1) * self.nx))
                A[0:self.nx, self.N * self.nx:(self.N + 1) * self.nx] = np.diag([1, 1, 1, 0])  # state <= ...
                A[self.nx: 2 * self.nx, self.N * self.nx:(self.N + 1) * self.nx] = -np.diag(
                    [1, 1, 1, 0])  # -state <= ...
                b = np.hstack((self.goal, -self.goal))
                self.term_set_A, self.term_set_b = A, b

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

    def terminal_constraint(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the constraint for the terminal set in this function
        A x <= b
            x = the state sequence with a variable length depending on the horizon
        :return: cvxpy compatible constraint
        """
        return self.term_set_A, self.term_set_b

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

        A = np.vstack(A)
        b = np.hstack(b)

        return A, b

    def LQR(self, x0) -> np.ndarray:
        """
        An unconstrained LQR controller that uses the unconstrained optimal control law.
        Mainly implemented for comparison reasons
        :param x0: the current state
        :return: the control input
        """
        return self.K @ (x0 - self.goal)

    def step(self, x0) -> np.ndarray:
        """
        Function to step the LQR controller if it should be used and otherwise raise an error
        :param x0: the current state
        :return: returns the first action of the optimal control sequence
        """
        if self.use_LQR:
            u0 = self.LQR(x0)
            u0 = np.clip(u0, self.input_lower, self.input_upper)
            self.stage_cost = (x0 - self.goal) @ self.Q @ (x0 - self.goal) + u0 @ self.R @ u0
            return u0
        else:
            raise NotImplementedError


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
        super().__init__(dt,
                         N,
                         lin_state,
                         lin_input,
                         terminal_constraint,
                         input_constraint,
                         state_constraint,
                         env,
                         )

        # Check lengths of constraints
        self.check_constraints()

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
        cost = problem.solve()
        if math.isinf(cost):
            # return [0, 0]
            raise OutsideTheRegionOfAttractionError

        # Save for visualisation purposes
        self.x_horizon = self.T @ x0 + self.S @ u.value
        self.x_horizon = self.x_horizon.reshape(-1, self.nx) + self.goal
        self.u_horizon = u.value.reshape(-1, 2)
        x_N = x.value[-4:]
        self.terminal_cost = x_N @ self.P @ x_N
        u0 = u.value[:self.nu]
        self.stage_cost = x0 @ self.Q @ x0 + u0 @ self.R @ u0

        return u.value[:self.nu]


class MPCOutputFB(MPC):
    def __init__(self,
                 dt: float,
                 N: int,
                 lin_state: Union[np.ndarray, list],
                 lin_input: Union[np.ndarray, list],
                 init_state: Union[np.ndarray, list],
                 terminal_constraint: bool = True,
                 input_constraint: bool = True,
                 state_constraint: bool = True,
                 env: BaseEnv = None,
                 ) -> None:
        """
        Initialize the MPC controller with output feedback. So the state is not measurable and the output is: y = [x, y]
        The system remains the same, but the state has to be derived from the output:
            x^+ = Ax + Bu
            y = Cx
            x_hat^+ = A x_hat + B u + L (y - C x_hat)
        :param dt: The timestep used to discretize the state space model
        :param N: The prediction horizon length used in the MPC controller
        :param lin_state: The state around which to linearize the state space model
        :param lin_input: The input around which to linearize the state space model
        :param init_state: The start state to initialize the observer
        :param env: The environment which also holds constraints
        """
        super().__init__(dt,
                         N,
                         lin_state,
                         lin_input,
                         terminal_constraint,
                         input_constraint,
                         state_constraint,
                         env,
                         )

        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])  # Observable

        # We use a reference output state
        self.y_goal = self.C @ env.goal

        # from scipy.linalg import solve_sylvester
        #
        # G = np.random.rand(3, 4)
        # # eigenvalue = 0.9
        # hat = np.diag([0.8, 0.7, -0.6, 0.8])
        # X = solve_sylvester(self.A.T, -hat, self.C.T @ G)
        # self.L = (G @ np.linalg.inv(X)).T
        self.L = np.array([[3.00000000e-01, 2.00284502e-16, 2.00000000e-01],
                           [3.48682242e+00, 1.90000000e+00, 3.86733776e-01],
                           [1.74341121e+00, 8.00000000e-01, 1.93366888e-01],
                           [-3.36822969e-16, -2.07305381e-16, 3.00000000e-01]])

        # print(self.L)
        self.x_estimate = np.array(init_state)
        self.previous_u = np.zeros(2)

        # Check lengths of constraints
        self.check_constraints()

    def optimal_target_selection(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the optimal target (x_ref, u_ref) for a certain y_ref
        :return: x_ref, u_ref; the reference state and input
        """
        xu_ref = cp.Variable(self.nx + self.nu)
        x_ref = xu_ref[:self.nx]
        u_ref = xu_ref[self.nx:]

        objective = cp.Minimize(cp.quad_form(x_ref, self.Q) + cp.quad_form(u_ref, self.R))
        constraints = []
        # Dynamics constraints
        A = np.hstack((np.vstack((np.eye(self.nx)-self.A, self.C)), np.vstack((-self.B, np.zeros((self.C.shape[0], self.B.shape[1]))))))
        b = np.hstack((np.zeros(self.nx), self.y_goal))
        constraints += [A @ xu_ref == b]
        # state constraints
        A, b = np.vstack(self.state_const_A), np.array(self.state_const_b)
        constraints += [A@x_ref <= b]
        # input constraints
        A, b = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.hstack((self.input_upper, -self.input_lower))
        constraints += [A@u_ref <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return x_ref.value, u_ref.value

    def luenberger_observer(self, y0: Union[np.ndarray, list[float]]):
        """
        Implementation fo the Luenberger_observer:
            x_hat^+ = A x_hat + B u + L (y - C x_hat)
            A - L C should be stable
        :param y0: The current output used to estimate the next state
        :param control_input: the input applied at the system
        :return: predicted new state
        """
        return self.A @ self.x_estimate + self.B @ self.previous_u + self.L @ (y0 - self.C @ self.x_estimate)

    def step(self, y0) -> np.ndarray:
        """
        Function to step the mpc controller
        :param y0: the current output (y = Cx), since the output cannot be measured
        :return: returns the first action of the optimal control sequence
        """
        x0 = self.luenberger_observer(y0)
        # print(x0)
        self.x_estimate = x0
        x_ref, u_ref = self.optimal_target_selection()

        u = cp.Variable((self.N * self.nu)) - np.hstack((u_ref,)*self.N)
        x = self.T @ x0 + self.S @ u
        x0 = x0 - x_ref
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
        cost = problem.solve()
        if math.isinf(cost):
            # return [0, 0]
            raise OutsideTheRegionOfAttractionError

        # Save for visualisation purposes
        self.x_horizon = self.T @ x0 + self.S @ u.value
        self.x_horizon = self.x_horizon.reshape(-1, self.nx) + self.goal
        self.u_horizon = u.value.reshape(-1, 2)
        self.cost = cost

        # Save values for the observer
        self.previous_u = u.value[:self.nu]

        return u.value[:self.nu]


class MPCOutputFBWithDisturbance(MPC):
    def __init__(self,
                 dt: float,
                 N: int,
                 lin_state: Union[np.ndarray, list],
                 lin_input: Union[np.ndarray, list],
                 init_state: Union[np.ndarray, list],
                 terminal_constraint: bool = True,
                 input_constraint: bool = True,
                 state_constraint: bool = True,
                 env: BaseEnv = None,
                 ) -> None:
        """
        Initialize the MPC controller with output feedback. So the state is not measurable and the output is: y = [x, y]
        There is also disturbance on the state and on the output, which results in the following model
            x^+ = A x + B u + Bd d (+ w)
            y = C x + Cd d (+ v)
            where d is a scalar
            The state is elongated to [x, d]^T but in the code the calculation of d is just added separately
        :param dt: The timestep used to discretize the state space model
        :param N: The prediction horizon length used in the MPC controller
        :param lin_state: The state around which to linearize the state space model
        :param lin_input: The input around which to linearize the state space model
        :param init_state: The start state to initialize the observer
        :param env: The environment which also holds constraints
        """
        super().__init__(dt,
                         N,
                         lin_state,
                         lin_input,
                         terminal_constraint,
                         input_constraint,
                         state_constraint,
                         env,
                         )
        self.C = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]])  # Observable
        # Much influence on y and psi (radians, so much more sensitive)
        self.Bd = np.array([0.001, 0.001, 0.001, 0.001])  # Not so much deviation on the states
        self.Cd = np.array([0.001, 0.01, 0.01])  # More deviation on the readings, especially in x direction
        self.Bd = np.array([1, 1, 1, 1])  # Not so much deviation on the states
        self.Cd = np.array([1, 1, 1])  # More deviation on the readings, especially in x direction
        self.C_hat = np.hstack((self.C, self.Cd.reshape(-1, 1)))

        # Observer gain for the state observer
        self.L1 = np.array([[3.00000000e-01, 2.00284502e-16, 2.00000000e-01],
                            [3.48682242e+00, 1.90000000e+00, 3.86733776e-01],
                            [1.74341121e+00, 8.00000000e-01, 1.93366888e-01],
                            [-3.36822969e-16, -2.07305381e-16, 3.00000000e-01]])
        # Observer gain for the disturbance estimation
        self.L2 = np.array([1, 1, 1])*1

        self.A_hat = np.vstack((np.hstack((self.A, self.Bd.reshape(-1, 1))), np.array([0, 0, 0, 0, 1]).reshape(1, -1)))
        self.B_hat = np.vstack((self.B, [0, 0]))

        # # ===================== ALGORITHM to find L =======================
        # from scipy.linalg import solve_sylvester
        #
        # G = np.random.rand(3, 5)
        # hat = np.diag([0.9, 0.9, -0.5, 0.9, 0.8])
        # X = solve_sylvester(self.A_hat.T, -hat, self.C_hat.T @ G)
        # L_hat = (G @ np.linalg.inv(X)).T
        # print(L_hat)
        # self.L1, self.L2 = L_hat[:-1], L_hat[-1]
        # # =============================== END =============================

        self.x_estimate = np.array(init_state)
        self.d_estimate = 0
        self.previous_u = np.zeros(2)

        # We use a reference output state
        self.y_goal = self.C @ env.goal

        # Check lengths of constraints
        self.check_constraints()

    def optimal_target_selection(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Determine the optimal target (x_ref, u_ref) for a certain y_ref
        :return: x_ref, u_ref; the reference state and input
        """
        xu_ref = cp.Variable(self.nx + self.nu)
        x_ref = xu_ref[:self.nx]
        u_ref = xu_ref[self.nx:]

        objective = cp.Minimize(cp.quad_form(x_ref, self.Q) + cp.quad_form(u_ref, self.R))
        constraints = []
        # Dynamics constraints
        A = np.hstack((np.vstack((np.eye(self.nx)-self.A, self.C)), np.vstack((-self.B, np.zeros((self.C.shape[0], self.B.shape[1]))))))
        b = np.hstack((self.Bd * self.d_estimate, self.y_goal - self.Cd * self.d_estimate))
        constraints += [A @ xu_ref == b]
        # state constraints
        A, b = np.vstack(self.state_const_A), np.array(self.state_const_b)
        constraints += [A@x_ref <= b]
        # input constraints
        A, b = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]), np.hstack((self.input_upper, -self.input_lower))
        constraints += [A@u_ref <= b]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return x_ref.value, u_ref.value

    def luenberger_observer(self, y0: Union[np.ndarray, list[float]]):
        """
        Implementation fo the Luenberger_observer:
            x_hat^+ = A x_hat + B u + L (y - C x_hat)
            A - L C should be stable
        :param y0: The current output used to estimate the next state
        :param control_input: the input applied at the system
        :return: predicted new state
        """
        x_new = self.A @ self.x_estimate + self.Bd * self.d_estimate + self.B @ self.previous_u + \
                self.L1 @ (y0 - self.C @ self.x_estimate - self.Cd * self.d_estimate)
        d_new = self.d_estimate + self.L2 @ (y0 - self.C @ self.x_estimate - self.Cd * self.d_estimate)
        print((y0 - self.C @ self.x_estimate))
        print(y0 - self.C @ self.x_estimate - self.Cd * self.d_estimate)
        return x_new, d_new

    def step(self, y0) -> np.ndarray:
        """
        Function to step the mpc controller
        :param y0: the current output (y = Cx), since the output cannot be measured
        :return: returns the first action of the optimal control sequence
        """
        x0, d0 = self.luenberger_observer(y0)

        print(f"d estimate: {self.d_estimate}")
        # x_ref, u_ref = self.optimal_target_selection()
        x_ref, u_ref = np.array([30, 1.5, 0, 0]), np.zeros(2)
        # print(f"x_ref: {x_ref}")
        # print(f"u_ref: {u_ref}")
        self.x_estimate = x0
        self.d_estimate = d0

        u = cp.Variable((self.N * self.nu)) - np.hstack((u_ref,)*self.N)
        to_stack = [np.vstack(
            [np.linalg.matrix_power(self.A, i - j) @ self.Bd if (i - j) >= 0 else np.zeros(4) for i in range(self.N)]).T
                    for j in range(self.N + 1)]
        to_stack.reverse()
        ABd = np.vstack(to_stack).sum(axis=1)
        # TODO: Add d to the state calculations -> Not necessary: disturbance adds same cost for every x
        x = self.T @ x0 + self.S @ u + ABd * self.d_estimate
        x0 = x0 - x_ref
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
        cost = problem.solve()
        if math.isinf(cost):
            # return [0, 0]
            raise OutsideTheRegionOfAttractionError

        # Save for visualisation purposes
        self.x_horizon = self.T @ x0 + self.S @ u.value
        self.x_horizon = self.x_horizon.reshape(-1, self.nx) + self.goal
        self.u_horizon = u.value.reshape(-1, 2)
        self.cost = cost

        # Save values for the observer
        self.previous_u = u.value[:self.nu]

        return u.value[:self.nu]
