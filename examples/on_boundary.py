import numpy as np
import os


def is_interior(point):
    point = np.array(point)
    interior = []
    for i in range(4):
        shifted_point = point
        shifted_point[i] += 0.001
        interior.append(np.all(A @ shifted_point <= b))
        shifted_point = point
        shifted_point[i] -= 0.001
        interior.append(np.all(A @ shifted_point <= b))

    return np.all(interior), interior


filename = 'RoadMultipleCarsEnv_30_1.5_0_0.npy'
directory = '../terminal_sets/'
file = os.path.join(directory, filename)
terminal_set = np.load(file)
A, b = terminal_set[..., :4], terminal_set[..., 4]

print(is_interior([29.8, 1.3, 0, 0]))
print(is_interior([30, 1.5, 0, 0]))
