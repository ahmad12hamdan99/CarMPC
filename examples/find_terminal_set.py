import numpy as np

from lib.environments import *
from lib.terminal_set import calc_terminal_set, visualise_set

environment = RoadOneCarEnv()
# environment.set_goal([30, 0, 0, 0])

# Calculate the terminal set
A, b = calc_terminal_set(environment, verbose=False)

# Visualise the terminal set
visualise_set(A, b, environment)

