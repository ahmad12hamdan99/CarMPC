import numpy as np

# Update timestep of the MPC; also time between different horizons. ZOH is used for the input
DT_CONTROL = 0.2  # s
# Update step of the simulator which uses the nonlinear continues time model
# the control input from the controller stays the same durin DT_CONTROL seconds
DT_SIMULATION = DT_CONTROL  # s
# Simulation update steps per MPC update steps
STEPS_UPDATE = int(DT_CONTROL / DT_SIMULATION)

# State and input around which to linearize
LINEARIZE_STATE = [0, 0, 0, 3]
LINEARIZE_INPUT = [0, 0]

# Time horizon length MPC
N = 20

# The matrices Q and R that work well; used for the MPC and LQR (terminal set calculations)
STAGE_COST_Q = np.diag([5, 5, 10, 10])
# STAGE_COST_R = np.diag([10, 100]) * 10
STAGE_COST_R = np.diag([10, 100])
