import numpy as np

array = np.empty((10, 4), dtype = np.float32)
array.fill(0)

i = 0
for i in range(9):
    i =+ 1
    for j in range(4):
        array[i, j] = i

print(array)