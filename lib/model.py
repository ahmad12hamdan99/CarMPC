import numpy as np


class CarTrailerDimension:
    l1 = 3.5
    l2 = 4
    l12 = 1
    car_length = 5
    car_width = 2
    trailer_length = 5.5
    trailer_width = 2
    triangle_length = 2


class CarTrailerModel:
    def __init__(self, v: float = 4, use_CT: bool = False, dt: float = 0.01) -> None:
        """
        The class with the model in it, which can be updated for every timestep.

        :param v: The constant speed of the car; forward defined as positive [m/s]
        :param use_CT: Whether to use the continuous time dynamics or the discretized system
        :param dt: The timestep to use in the continuous time dynamics
        """
        self.l1 = CarTrailerDimension.l1
        self.l2 = CarTrailerDimension.l2
        self.l12 = CarTrailerDimension.l12
        self.v = v
        # state = [x, y, psi, gamma]
        self.state = np.zeros(4)
        self.use_CT = use_CT
        self.dt = dt if use_CT else None

    def reset(self, state: np.ndarray = np.zeros(4)) -> None:
        self.state = state

    def dynamics_discrete(self, state: np.ndarray, delta_f: float) -> np.ndarray:
        """
        Contains the system dynamics -> can be a non-linear model
        should be a DISCRETE TIME model.
        :param state: The current state of the system
        :param delta_f: The steering angle in RADIANS
        :return: None
        """
        raise NotImplementedError
        # new_state = state + delta_f
        # return new_state

    def dynamics_continuous(self, state: np.ndarray, delta_f: float) -> np.ndarray:
        """
        Contains the system dynamics -> can be a non-linear model
        should be a CONTINUOUS TIME model.
        :param state: The current state of the system
        :param delta_f: The steering angle in RADIANS
        :return: None
        """
        x, y, psi, gamma = state
        u = np.tan(delta_f)
        dx = self.v * np.cos(psi)
        dy = self.v * np.sin(psi)
        dpsi = self.v / self.l1 * u
        dgamma = (self.v / self.l1 + (self.v * self.l12) / (self.l1 * self.l2) * np.cos(
            gamma)) * u - self.v / self.l2 * np.sin(gamma)

        return np.array([dx, dy, dpsi, dgamma]) * self.dt + state

    def get_log(self, delta_f) -> dict[str, np.ndarray]:
        """
        Creates a log of the state, inputs and calculates the position and orientation of the trailer
        :param delta_f: The steering angle in radians
        :return: dict{'car': [x_car, y_car, psi_car, gamma], 'trailer': [x_tr, y_tr, psi_tr, gamma], 'inputs': [delta_f, v]}
        """
        psi_car = self.state[2]
        gamma = self.state[3]
        P_car = self.state[:2]
        P_towbar = P_car - np.array([np.cos(psi_car), np.sin(psi_car)]) * self.l12
        psi_tr = psi_car - gamma
        P_trailer = P_towbar - np.array([np.cos(psi_tr), np.sin(psi_tr)]) * self.l2

        log = dict()
        log['car'] = self.state
        log['trailer'] = np.hstack((P_trailer, psi_tr, gamma))
        log['inputs'] = np.array([delta_f, self.v])
        return log

    def step(self, delta_f) -> dict[str, np.ndarray]:
        """
        Updates the state with the input commands
        :param delta_f: The steering angle in RADIANS
        :return: The log data [state, inputs] = [x, y, psi, v, delta_f, v]
        """
        # TODO: Check input constraints on delta_f
        if self.use_CT:
            self.state = self.dynamics_continuous(self.state, delta_f)
        else:
            self.state = self.dynamics_discrete(self.state, delta_f)

        return self.get_log(delta_f)
