from scipy.sparse import random
from scipy.io import mmwrite
import numpy as np

m = 1600
rows, cols = m, m
density = 0.1

def scaled_uniform(size):
    return np.random.uniform(0.1, 5.0, size)

A = random(rows, cols, density=density, format='coo', dtype='float64', data_rvs=scaled_uniform)
mmwrite(f"sp_mat_{m}.mtx", A)
