import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Random Seed Fixed
torch.manual_seed(777)

# GPU 사용 가능할 경우 Seed 고정
if device == 'cuda':
    torch.cuda.manaul_seed_all(777)

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
            torch.nn.Relu(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        # Second Layer
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.Relu(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        # 전결합층
        self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias = True)

        # 전결합층 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.view(out.size(0), -1)    # Flatten