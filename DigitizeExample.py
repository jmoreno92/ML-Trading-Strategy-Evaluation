import pandas as pd
import numpy as np

x = np.array([-10,0.2, 6.4, 3.0, 1.6,20])
bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
inds = np.digitize(x, bins)
print(inds)

# for n in range(x.size):
#     print(bins[inds[n] - 1], "<=", x[n], "<", bins[inds[n]])