import numpy as np

from lib.matrix_gen import costgen, predmod
from lib.mpc import MPC

# Linearize the model
eq_state = np.array([0, 0, 0, 0, 3])  # Assume a velocity of 3 m/s in the state around which we linearized
eq_input = np.array([0, 0])
A_lin, B_lin = MPC.linearized_model(eq_state, eq_input)
# Discretize the model
A, B = MPC.discretized_model(A_lin, B_lin, dt=0.1)  # discretize with a timestep of 0.1 s. Might be a bit small
N = 10
Q = np.diag([3, 3, 1, 1, 1])
R = np.diag([0.2, 0.2])
P = np.diag(np.arange(5))

T, S = predmod(A, B, N)
H, h, _ = costgen(Q, R, P, T, S, nx=5)

print(f"A:\n{A}\nB:\n{B}")
print(f"T:\n{T}\nS:\n{S}")
print(f"H:\n{H}\nh:\n{h}")


