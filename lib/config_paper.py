# Update timestep of the MPC; also time between different horizons. ZOH is used for the input
DT_CONTROL = 0.2  # s
# Update step of the simulator which uses the nonlinear continues time model
DT_SIMULATION = 0.01  # s
# Simulation update steps per MPC update steps
STEPS_UPDATE = int(DT_CONTROL / DT_SIMULATION)

# State and input around which to linearize
LINEARIZE_STATE = [0, 0, 0, 3]
LINEARIZE_INPUT = [0, 0]

# Time horizon length MPC
N = 40
