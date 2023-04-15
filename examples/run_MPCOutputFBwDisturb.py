import numpy as np
import itertools

from lib.mpc import MPCOutputFBWithDisturbance
from lib.simulator import CarSimulator, CarTrailerDimension
from lib.visualize_state import plot_live, plot_history, plot_graphs
from lib.environments import *
# Contains the all CAPS variables
from lib.configuration import *

# Set up the environment
environment = RoadEnv()
start_position = [5, -1.5, 0, 0]

# Set up the MPC controller
controller = MPCOutputFBWithDisturbance(dt=DT_CONTROL, N=N, lin_state=LINEARIZE_STATE, lin_input=LINEARIZE_INPUT,
                                        init_state=start_position, terminal_constraint=True, input_constraint=True,
                                        state_constraint=True, env=environment)
controller.set_goal(environment.goal)

# Set up the simulation environment (uses the same non-linearized model with a smaller timestep
simulation = CarSimulator(dt=DT_SIMULATION, C=controller.C, Cd=controller.Cd)
simulation.reset(np.array(start_position))

control_input = [0, 0]
max_gamma = 0
max_pred_gamma = 0
car_states, inputs, costs = [], [], []
for i in itertools.count():

    log = simulation.step(control_input, d=0)
    car_states.append(log['car'])
    inputs.append(log['inputs'])
    costs.append(controller.cost)

    if i % STEPS_UPDATE == 0:
        control_input = controller.step(simulation.output)
        print(simulation.state - controller.x_estimate)
        print("="*50)
        plot_live(log, dt=0.1, time=simulation.time, goal=controller.x_estimate, state_horizon=controller.x_horizon,
                  env=environment)

    if np.all(np.abs(simulation.state - controller.goal) <= 1e-1).astype(bool) or i > 30 / DT_SIMULATION:
        break
        # simulation.reset(np.array(start_position))

log_history = {'car': np.array(car_states), 'inputs': np.array(inputs), 'costs': np.array(costs)}
plot_history(log_history, dt=DT_SIMULATION, goal=controller.goal, env=environment)
plot_graphs(log_history, dt=DT_SIMULATION)
