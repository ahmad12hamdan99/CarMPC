import numpy as np

from lib.model import CarTrailerModel, CarTrailerDimension
from lib.visualize_state import plot_graph

model_ct = CarTrailerModel(use_CT=True, dt=0.01)

car_states_ct, trailer_states_ct, inputs_ct = [], [], []
for i in range(3000):
    if i > 2000:
        delta = -0.3
    elif i > 1000:
        delta = 0
    else:
        delta = 0.3

    log = model_ct.step(delta)
    car_states_ct.append(log['car'])
    trailer_states_ct.append(log['trailer'])
    inputs_ct.append(log['inputs'])

car_states_ct = np.array(car_states_ct)
trailer_states_ct = np.array(trailer_states_ct)
inputs_ct = np.array(inputs_ct)

log_ct = {'car': car_states_ct, 'trailer': trailer_states_ct, 'inputs': inputs_ct}
print(log_ct['car'][-1])

plot_graph(log_ct, dt=0.01)

