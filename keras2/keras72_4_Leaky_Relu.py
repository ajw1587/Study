import numpy as np
import matplotlib.pyplot as plt

def Leaky_Relu(x):
    return np.maximum(0.01 * x, x)

x = np.arange(-5, 5, 0.1)
y = Leaky_Relu(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()