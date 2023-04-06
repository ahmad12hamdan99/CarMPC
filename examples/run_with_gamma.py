import numpy as np
import itertools

from lib.mpc import MPC
from lib.simulator import CarTrailerSimWithAcc, CarTrailerDimension
from lib.visualize_state import plot_live

# Simulation runs at a higher rate than the MPC controller
DT_CONTROL = 0.1  # s
DT_SIMULATION = 0.01  # s
STEPS_UPDATE = int(0.1 / 0.01)

# Set up the MPC controller
# [0, 0, 0.1, -0.1, -1]
controller = MPC(dt=DT_CONTROL, N=20, lin_state=[0, 0, 0, 0, -2], lin_input=[0, 0], terminal_constraint=False, input_constraint=True, state_constraint=True)
controller.set_goal([-20, -1, 0, 0, 0])

# Set up the simulation environment (uses the same non-linearized model with a smaller timestep
simulation = CarTrailerSimWithAcc(dt=DT_SIMULATION)
# start_position = [0, 5, 0.3, 0.05, 0]
start_position = [20, 0, 0, 0, 0]
simulation.reset(np.array(start_position))

control_input = [0, 0]
for i in itertools.count():
    log = simulation.step(control_input)
    if i % STEPS_UPDATE == 0:
        control_input = controller.step(simulation.state)
        print(f"input: {control_input}")
        print(f"state: {simulation.state}")
        print("=" * 30)
        plot_live(log, dt=0.01, time=simulation.time, goal=controller.goal, state_horizon=controller.x_horizon, lim=[(-30, 30), (-10, 30)])

    if np.all(np.abs(simulation.state - controller.goal) <= 1e-2).astype(bool):
        simulation.reset(np.array(start_position))

