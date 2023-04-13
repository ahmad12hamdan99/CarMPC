import numpy as np
import itertools

from lib.mpc import MPCStateFB
from lib.simulator import CarTrailerSimWithAcc, CarTrailerDimension
from lib.visualize_state import plot_live, plot_history, plot_graphs
from lib.environments import *

# Simulation runs at a higher rate than the MPC controller
DT_CONTROL = 0.1  # s
DT_SIMULATION = 0.01  # s
STEPS_UPDATE = int(DT_CONTROL / DT_SIMULATION)
LINEARIZE_STATE = [0, 0, 0, 3]
LINEARIZE_INPUT = [0, 0]

# Set up the environment
environment = RoadEnv()
environment.set_goal([0, 0, 0, 0])
environment.set_lim([(-50, 10), (-10, 10)])

# Set up the LQR controller
controller = MPCStateFB(dt=DT_CONTROL, N=40, lin_state=LINEARIZE_STATE, lin_input=LINEARIZE_INPUT, env=environment)
controller.set_goal(environment.goal)

# Set up the simulation environment (uses the same non-linearized model with a smaller timestep)
simulation = CarTrailerSimWithAcc(dt=DT_SIMULATION, clip=True)
start_position = [-50, 0, 0, 0]
simulation.reset(np.array(start_position))

control_input = [0, 0]
car_states, inputs = [], []
for i in itertools.count():
    log = simulation.step(control_input)
    car_states.append(log['car'])
    inputs.append(log['inputs'])

    if i % STEPS_UPDATE == 0:
        control_input = controller.step(simulation.state, LQR=True)
        plot_live(log, dt=0.01, time=simulation.time, goal=controller.goal, state_horizon=controller.x_horizon,
                  env=environment)

    if np.all(np.abs(simulation.state - controller.goal) <= 1e-1).astype(bool) or i > 30 / DT_SIMULATION:
        break
        # simulation.reset(np.array(start_position))

log_history = {'car': np.array(car_states), 'inputs': np.array(inputs)}
plot_history(log_history, dt=DT_SIMULATION, goal=controller.goal, env=environment)
plot_graphs(log_history, dt=DT_SIMULATION)
