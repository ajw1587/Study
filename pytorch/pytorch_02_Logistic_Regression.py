import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

# cost Function: (H(x),y)=−[ylogH(x)+(1−y)log(1−H(x))]
# https://wikidocs.net/57805

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

torch.Size([6, 2])
torch.Size([6, 1])

# 모델 초기화
W = torch.zeros((2, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)

# optimizer 설정
optimizer = torch.optim.SGD([W, b], lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs):
  # Cost 계산
  # torch에서 sigmoid 함수 지원
  hypothesis2 = torch.sigmoid(x_train.matmul(W) + b)
  cost = -(y_train * torch.log(hypothesis2) + 
             (1 - y_train) * torch.log(1 - hypothesis2)).mean()

  # cost로 H(x) 개선
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  if epoch % 100 == 0:
    print('Epoch {:4d}/{}     Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

# 예측값 출력
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)