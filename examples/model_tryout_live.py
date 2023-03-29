import matplotlib.pyplot as plt
import numpy as np
import time
from lib.simulator import CarTrailerSim, CarTrailerDimension
from lib.visualize_state import plot_live

model_ct = CarTrailerSim(use_CT=True, v=-4, dt=0.01)
model_ct.reset(np.array([0, 0, np.pi, -0.1]))
# plt.figure()
for i in range(3000):
    delta = -0.05
    log = model_ct.step(delta)
    if i % 10 == 0:
        plot_live(log, dt=0.001)

