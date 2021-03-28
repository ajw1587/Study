# https://wikidocs.net/63618

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random Seed Fixed
torch.manual_seed(777)

# GPU 사용 가능할 경우 Seed 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root = 'MNIST_data/',
                          train = True,
                          transform = transforms.ToTensor(),
                          download = True)

mnist_test = dsets.MNIST(root = 'MNIST_data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)


# Data Loader
data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last = True)

# Model 설계
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # First Layer
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        # Second Layer
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        # 전결합층
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias = True)

        # 전결합층 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)    # Flatten
        out = self.fc(out)
        return out

model = CNN().to(device)

# 비용 함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# 총 배치의 수 출력
total_batch = len(data_loader)
print('총 배치수: {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0
    # i = 0

    for X, Y in data_loader:                # X는 미니 배치, Y는 레이블 600번 반복

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost/total_batch        # 600번 반복으로 인한 나누기
        # print(i)
        # i += 1
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    
# TEST
with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())