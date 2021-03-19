import torch
import torch.nn as nn

# 배치 크기 x 채널 x 높이(height) x 너비(width)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기: ', inputs.shape)

conv1 = nn.Conv2d(1, 32, 3, padding = 1)
print(conv1)

conv2 = nn.COnv2d(32, 64, kernel_size = 3, padding = 1)
print(conv2)

# MaxPool 사용시 정수 하나만 인자로 넣으면
# kernel_size, stride 둘다 해당값으로 지정
pool = nn.MaxPool2d(2)
print(pool)

