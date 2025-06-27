import numpy as np

m = 1600
n = 128

# Scale rand output from [0,1) to [0.1, 5]
B = 0.1 + (5.0 - 0.1) * np.random.rand(m, n)
B = B.astype(np.float64)

np.savetxt(f"dense_mat_{m}_{n}.csv", B, delimiter=",")
