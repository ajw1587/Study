import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 모델
model = nn.Sequential(
    nn.Linear(2, 1),    # input = 2, output = 1
    nn.Sigmoid()
)

# Optimizer 설정
optimizer = optim.SGD(model.parameters(), lr = 1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

  # H(x) 계산
  hypothesis = model(x_train)

  # cost 계산
  cost = F.binary_cross_entropy(hypothesis, y_train)

  # cost로 H(x) 개선
  optimizer.zero_grad()
  cost.backward()
  optimizer.step()

  # 20번마다 로그 출력
  if epoch % 10 == 0:
    prediction = hypothesis >= torch.FloatTensor([0.5])    # 예측값이 0.5 넘을 경우 True
    correct_prediction = prediction.float() == y_train     # 실제값과 예측값이 일치하는 경우 True
    
    print(prediction)
    print(correct_prediction)

    accuracy = correct_prediction.sum().item() / len(correct_prediction)

    print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
            epoch, nb_epochs, cost.item(), accuracy * 100))

# 훈련후의 W, b
print(list(model.parameters()))
# [Parameter containing:
# tensor([[3.2534, 1.5181]], requires_grad=True), Parameter containing:
# tensor([-14.4839], requires_grad=True)]