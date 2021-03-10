import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(77)
if device == 'cuda':
    torch.cuda.manual_seed_all(77)

x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

model = nn.Sequential(
    nn.Linear(2, 10, bias = True),
    nn.Sigmoid(),
    nn.Linear(10, 10, bias = True),

    nn.Sigmoid(),
    nn.Linear(10, 10, bias = True),

    nn.Sigmoid(),
    nn.Linear(10, 1, bias = True),
    nn.Sigmoid()
).to(device)

# Cost, Optimizer
criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 1)
# from 0.1 to 1

for epoch in range(10001):
    optimizer.zero_grad()
    hypothesis = model(x)

    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, cost.item())


with torch.no_grad():
    hypothesis = model(x)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y).float().mean()

    print('모델의 출력값(Hypothesis): ', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
    print('실제값(y): ', y.cpu().numpy())
    print('정확도(Accuracy): ', accuracy.item())
