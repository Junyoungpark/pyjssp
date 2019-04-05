import numpy as np


def jssp_sampling(m, n, low=5, high=100):
    machine_mat = np.ndarray(shape=(n, m))
    process_time_mat = np.random.randint(low, high, size=(n, m))
    for i in range(n):
        machine_mat[i] = np.random.permutation(m)
    return machine_mat, process_time_mat
