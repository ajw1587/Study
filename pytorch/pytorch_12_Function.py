import numpy as np
import matplotlib.pyplot as plt

# 1. sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.plot([0,0], [1.0, 0.0], ':')
plt.title('Sigmoid Function')
plt.show()

# 2. Tanh
x = np.arange(-5.0, 5.0, 0.1)
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0,0], [1.0, -1.0], ':')
plt.axhline(y = 0, color = 'orange', linestyle = '--')
plt.title('Tanh Function')
plt.show()

# 종류에 따른 Function
# 1. binary data: nn.BCELoss()
# 2. categorical data: nn.CrossEntropyLoss()
# 3. Linear: MSE