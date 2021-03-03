import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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
 
# https://wikidocs.net/60324