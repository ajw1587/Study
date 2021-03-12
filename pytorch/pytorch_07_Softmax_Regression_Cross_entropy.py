import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 1. Data
x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
# print(x_train.shape)    # torch.Size([8, 4])
# print(y_train.shape)    # torch.Size([8])

# 2. One_hot_encoding
# y_one_hot = torch.zeros(8, 3)
# y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
# print(y_one_hot.shape)  # torch.Size([8, 3])

# 3. Model
W = torch.zeros((4, 3), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
optimizer = optim.SGD([W, b], lr = 0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # Cost
    z = x_train.matmul(W) + b
    cost = F.cross_entropy(z, y_train)

    # Cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{}      Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))