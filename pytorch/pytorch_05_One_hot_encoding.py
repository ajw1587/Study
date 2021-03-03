
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

z = torch.rand(3, 5, requires_grad = True)
print(z)
print('=========================================')

hypothesis = F.softmax(z, dim = 1)      # 행 기준으로 value들의 합이 1
print(hypothesis)
print('=========================================')

y = torch.randint(5, (3,)).long()       # 0~4 사이의 숫자로 (3, ) 배열 생성
print(y)
print('=========================================')

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)    # 행(dim = 1) 기준으로 y.unsqueeze(1)위치의 값을 1로 바꿔준다.
print(y_one_hot)
# print(y.unsqueeze(1))