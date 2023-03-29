import numpy as np
import itertools

from lib.mpc import MPC
from lib.simulator import CarTrailerSimWithAcc, CarTrailerDimension
from lib.visualize_state import plot_live

# Simulation runs at a higher rate than the MPC controller
DT_CONTROL = 1  # s
DT_SIMULATION = 0.01  # s
STEPS_UPDATE = int(0.1 / 0.01)

# Set up the MPC controller
controller = MPC(dt=DT_CONTROL, N=10, lin_state=[0, 0, 0, 0, -3], lin_input=[0, 0])
controller.set_goal([-30, 1, 0, 0, 0])

# Set up the simulation environment (uses the same non-linearized model with a smaller timestep
simulation = CarTrailerSimWithAcc(dt=DT_SIMULATION)
simulation.reset(np.array([-5, 0, 0, 0, 0]))

control_input = [0, 0]

for i in itertools.count():
    log = simulation.step(control_input)
    if i % STEPS_UPDATE == 0:
        control_input = controller.step(simulation.state)
        plot_live(log, dt=0.01, time=simulation.time, goal=controller.goal, state_horizon=controller.x_horizon)

    if np.all(np.abs(simulation.state - controller.goal) <= 1e-2).astype(bool):
        simulation.reset(np.array([-5, 0, 0, 0, 0]))
