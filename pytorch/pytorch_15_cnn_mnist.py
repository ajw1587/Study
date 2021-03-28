# https://wikidocs.net/63618

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random Seed Fixed
torch.manual_seed(777)

# If GPU is avaliable, Fix Random Seed
if device == 'cuda':
    torch.cuda.manual_seed(777)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Data
mnist_train = dsets.MNIST(root = 'MNIST_data/',
                          train = True,
                          transform = transforms.ToTensor(),
                          download = True)

mnist_test = dsets.MNIST(root = 'MNIST_data/',
                         train = False,
                         transform = transforms.ToTensor(),
                         download = True)

data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          drop_last = True)

print(len(data_loader))