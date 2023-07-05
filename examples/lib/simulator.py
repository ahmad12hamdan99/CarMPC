import numpy as np
from typing import Union


class CarTrailerDimension:
    l1 = 3.5
    l2 = 4
    l12 = 1
    car_length = 5
    car_width = 2
    trailer_length = 5.5
    trailer_width = 2
    triangle_length = 2


class CarSimulator:
    def __init__(self, dt: float = 0.01, clip: bool = False, C: np.ndarray = None, Cd: np.ndarray = None) -> None:
        """
        The class with the model in it, which can be updated for every timestep.

        :param dt: The timestep to use in the continuous time dynamics
        :param clip: if true, clip the control input instead of throwing an error
        :param C: The matrix to obtain the output for a state space model; y = Cx
        :param Cd: the matrix to add disturbance to the output; y = C x + Cd d
        """
        self.time = 0
        self.l1 = CarTrailerDimension.l1
        self.l2 = CarTrailerDimension.l2
        self.l12 = CarTrailerDimension.l12
        # state = [x, y, psi, v]
        self.state = np.zeros(4)
        self.dt = dt
        self.clip = clip
        self.C = np.eye(4) if C is None else C
        assert self.C.shape[1] == 4, f"The length of the second dimension (currently {C.shape[1]}) of C should be equal to the state length (4)."
        self.output = self.C @ self.state
        self.Cd = Cd
        # Constraints --> Need to set some meaningful values
        self.x_lower, self.x_upper = -np.inf, np.inf  # m
        self.y_lower, self.y_upper = -np.inf, np.inf  # m
        self.psi_lower, self.psi_upper = -np.pi, np.pi  # rad
        self.v_lower, self.v_upper = -10, 10  # m/s
        # Add some margins
        self.delta_lower, self.delta_upper = -np.pi / 4 - 0.05, np.pi / 4 + 0.05  # rad
        self.acc_lower, self.acc_upper = -2 - 0.05, 2 + 0.05  # m/s^2

    def reset(self, state: Union[np.ndarray, list] = np.zeros(5)) -> None:
        assert np.array(state).shape == (4,), f"The state should have shape (4,), but has shape{np.array(state).shape}"
        self.state = state

    def dynamics_continuous(self, state: np.ndarray, control_input: Union[np.ndarray, list]) -> np.ndarray:
        """
        Contains the system dynamics -> can be a non-linear model
        should be a CONTINUOUS TIME model.
        :param state: The current state of the system
        :param control_input: A 1D array with to elements: [acceleration, steering angle]
        :return: None
        """
        assert np.array(control_input).shape == (2,),\
            f"The input should have shape (2,), but has shape{np.array(control_input).shape}"
        x, y, psi, v = state
        a, delta_f = control_input
        u = np.tan(delta_f)
        dx = v * np.cos(psi)
        dy = v * np.sin(psi)
        dpsi = v / self.l1 * u
        dv = a

        return np.array([dx, dy, dpsi, dv]) * self.dt + state

    def get_log(self, control_input: Union[np.ndarray, list]) -> dict[str, np.ndarray]:
        """
        Creates a log of the state, inputs and calculates the position and orientation of the trailer
        :param control_input: A 1D array with to elements: [acceleration, steering angle]
        :return: dict{'car': [x_car, y_car, psi_car, gamma], 'trailer': [x_tr, y_tr, psi_tr, gamma], 'inputs': [delta_f, v]}
        """
        a, delta_f = control_input
        log = dict()
        log['car'] = self.state
        log['inputs'] = np.array([a, delta_f])
        return log

    def check_input(self, control_input: Union[np.ndarray, list[float]]) -> Union[np.ndarray, list]:
        """
        Throws an assert if the input is outside the input constraints
        :param control_input: A 1D array with to elements: [acceleration, steering angle]
        """
        a, delta_f = control_input
        if self.clip:
            control_input = np.clip(control_input, [self.acc_lower, self.delta_lower], [self.acc_upper, self.delta_upper])
        else:
            assert np.all([a, -a] <= [self.acc_upper, -self.acc_lower]), \
                f"Acceleration should be between [{self.acc_lower, self.acc_upper}], but is {a}."
            assert np.all([delta_f, -delta_f] <= [self.delta_upper, -self.delta_lower]), \
                f"Steering angle should be between [{self.delta_lower:.3f}, {self.delta_upper:.3f}], but is {delta_f}."

        return control_input

    def step(self, control_input: Union[np.ndarray, list], d: float = None) -> dict[str, np.ndarray]:
        """
        Updates the state with the input commands
        :param control_input: A 1D array with to elements: [acceleration, steering angle]
        :param d: The disturbance on the output (y = C x + Cd d)
        :return: The log data [state, inputs] = [x, y, psi, v, delta_f, v]
        """
        # TODO: Check input constraints on delta_f and acceleration
        # TODO: Check constraints on state
        # Check if the control input statisfy the constraints
        self.check_input(control_input)

        self.state = self.dynamics_continuous(self.state, control_input)

        self.output = self.C @ self.state
        if self.Cd is not None and d is not None:
            self.output += self.Cd * d

        self.time += self.dt
        return self.get_log(control_input)
