import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import random
import matplotlib.pyplot as plt

from six.moves import urllib    
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# GPU 연산
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
print('다음 장치로 학습합니다: ', device)

# RandomSeed Fix
torch.manual_seed(777)
random.seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# hyperparameters
training_epochs = 15
batch_size = 100

# MNIST Dataset
mnist_train = dsets.MNIST(root = 'MNIST_data/',
                          train = True,
                          transform = transforms.ToTensor(),
                          download = True)
# root: 데이터의 경로
# train: 테스트용 데이터를 가져올지 학습용데이터를 가져올지 표시, True: Train용
# transform: 에디터의 형태, transforms.ToTensor() -> Tensor형태
# download: True일시 MNIST 데이터가 없으면 다운을 받는다.

mnist_test = dsets.MNIST(root = 'MNIST_data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)

data_loader = DataLoader(dataset = mnist_train,
                         batch_size = batch_size,
                         shuffle = True,
                         drop_last = True)

# Model
hypothesis = nn.Linear(784, 10, bias = True).to(device)     # to()는 어디서 연산을 수행할지

# Cost Function
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(hypothesis.parameters(), lr = 0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        # 배치 크기가 100이므로 X의 shape는 (100, 784)
        x = X.view(-1, 28*28).to(device)
        y = Y.to(device)

        optimizer.zero_grad()
        cost = criterion(hypothesis(x), y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    
    print('Epoch: ', '%04d' % (epoch + 1), 'Cost = ', '{:.9f}'.format(avg_cost))

print('Learning finished')