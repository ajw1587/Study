import numpy as np

array = np.empty((10, 4), dtype = np.float32)
array.fill(0)

array[5, 0] = 5
print(array)