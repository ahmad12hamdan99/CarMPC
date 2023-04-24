import numpy as np
import itertools

from lib.mpc import MPCOutputFB
from lib.simulator import CarSimulator, CarTrailerDimension
from lib.visualize_state import plot_live, plot_history, plot_graphs
from lib.environments import *
# Contains the all CAPS variables
from lib.configuration import *

# Set up the environment
environment = RoadMultipleCarsEnv()
start_position = [5, -1.5, 0, 0]

# Set up the MPC controller
controller = MPCOutputFB(dt=DT_CONTROL, N=N, lin_state=LINEARIZE_STATE, lin_input=LINEARIZE_INPUT,
                         init_state=start_position, terminal_constraint=True, input_constraint=True,
                         state_constraint=True, env=environment)
controller.set_goal(environment.goal)

# Set up the simulation environment (uses the same non-linearized model with a smaller timestep
simulation = CarSimulator(dt=DT_SIMULATION, C=controller.C)
simulation.reset(np.array(start_position))

control_input = [0, 0]
max_gamma = 0
max_pred_gamma = 0
car_states_sim, car_states_est, inputs = [], [], []
for i in itertools.count():
    log = simulation.step(control_input)
    inputs.append(log['inputs'])

    if i % STEPS_UPDATE == 0:
        control_input = controller.step(simulation.output)
        plot_live(log, dt=0.1, time=simulation.time, estimate=controller.x_estimate, state_horizon=controller.x_horizon,
                  env=environment)

    car_states_sim.append(log['car'])
    car_states_est.append(controller.x_estimate)

    if i >= 15 / DT_SIMULATION:
        break
        # simulation.reset(np.array(start_position))

car_states_sim = np.array(car_states_sim)
car_states_est = np.array(car_states_est)
log_history = {'car': np.array(car_states_sim), 'inputs': np.array(inputs), 'estimate': np.array(car_states_est)}
plot_history(log_history, dt=DT_SIMULATION, goal=controller.goal, env=environment)
plot_graphs(log_history, dt=DT_SIMULATION)


