import numpy as np
import itertools

from lib.mpc import MPC
from lib.simulator import CarTrailerSimWithAcc, CarTrailerDimension
from lib.visualize_state import plot_live, plot_history, plot_graphs

# Simulation runs at a higher rate than the MPC controller
DT_CONTROL = 0.2  # s
DT_SIMULATION = 0.01  # s
STEPS_UPDATE = int(DT_CONTROL / DT_SIMULATION)
# STEPS_UPDATE = 5
# Set up the MPC controller
# [0, 0, 0.1, -0.1, -1]
controller = MPC(dt=DT_CONTROL, N=40, lin_state=[0, 0, 0, 0, 3], lin_input=[0, 0], terminal_constraint=True,
                 input_constraint=True, state_constraint=True)
controller.set_goal([30, 1.5, 0, 0, 0])

# Set up the simulation environment (uses the same non-linearized model with a smaller timestep
simulation = CarTrailerSimWithAcc(dt=DT_SIMULATION)
start_position = [5, -1.5, 0, 0, 0]
simulation.reset(np.array(start_position))

control_input = [0, 0]
max_gamma = 0
max_pred_gamma = 0
car_states, trailer_states, inputs = [], [], []
for i in itertools.count():
    log = simulation.step(control_input)
    car_states.append(log['car'])
    trailer_states.append(log['trailer'])
    inputs.append(log['inputs'])

    if i % STEPS_UPDATE == 0:
        print(f"Speed: {log['car'][4]}")
        control_input = controller.step(simulation.state)
        pred_gamma = np.max(controller.x_horizon[:, 3])
        if pred_gamma > max_pred_gamma:
            max_pred_gamma = pred_gamma
            print(f"new max pred gamma: \t{max_pred_gamma}")
        if simulation.state[3] > max_gamma:
            max_gamma = simulation.state[3]
            print(f"new max gamma: \t{max_gamma}")
        # print(f"input: {control_input}")
        # print(f"state: {simulation.state}")
        # print("=" * 30)
        plot_live(log, dt=0.01, time=simulation.time, goal=controller.goal, state_horizon=controller.x_horizon,
                  lim=[(-10, 50), (-15, 15)])

    if np.all(np.abs(simulation.state - controller.goal) <= 1e-1).astype(bool):
        break
        # simulation.reset(np.array(start_position))

log_history = {'car': np.array(car_states), 'trailer': np.array(trailer_states), 'inputs': np.array(inputs)}
# plot_history(log_history, dt=DT_SIMULATION)
plot_graphs(log_history, dt=DT_SIMULATION)
