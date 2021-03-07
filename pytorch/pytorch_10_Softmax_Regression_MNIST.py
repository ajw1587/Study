import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
from torch.utils.data import DataLoader

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import random
import matplotlib.pyplot as plt

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
# transform: 에디터의 형태, pytorch의 0~1, (C, H, W)형태로 변경
# download: True일시 MNIST 데이터가 없으면 다운을 받는다.

mnist_test = dsets.MNIST(root = 'MNIST_data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)

data_loader = DataLoader(dataset = minst_train,
                         batch_size = batch_size,
                         shuffle = True,
                         drop_last = True)

# Model
linear = nn.Linear(784, 10, bias = True).to(device)     # to()는 어디서 연산을 수행할지

# for i , (images, labels) in enumerate(data_loader):
#     print(type(images))
#     print(labels)