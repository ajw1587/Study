import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(77)
if device == 'cuda':
    torch.cuda.manual_seed_all(77)

x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear = nn.Linear(2, 1, bias = True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear, sigmoid).to(device)

# Cost and Optimizer
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(x)

    # Cost
    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())


with torch.no_grad():
    hypothesis = model(x)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y).float().mean()

    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(y): ', y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())

# 모델의 출력값(Hypothesis):  [[0.5]
#  [0.5]
#  [0.5]
#  [0.5]]
# 모델의 예측값(Predicted):  [[0.]
#  [0.]
#  [0.]
#  [0.]]
# 실제값(y):  [[0.]
#  [1.]
#  [1.]
#  [0.]]
# 정확도(Accuracy):  0.5