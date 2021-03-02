import numpy as np
import matplotlib.pyplot as plt

alp = 2

def Selu(x):
    return (x > 0)*x + (x <= 0) * (alp*(np.exp(x) -1))

x = np.arange(-5, 5, 0.1)
y = Selu(x)

print(x)
print(y)

plt.plot(x, y)
plt.grid()
plt.show()