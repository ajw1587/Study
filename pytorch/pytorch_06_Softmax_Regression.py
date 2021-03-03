import torch
import torch.nn.functional as F

torch.manual_seed(1)

# 1. softmax
z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim = 0)
# print(hypothesis)

# 2. softmax
z = torch.rand(3, 5, requires_grad = True)
hypothesis = F.softmax(z, dim = 1)
print(hypothesis)

# y 값 생성
y = torch.randint(5, (3, ))
print(y)

# y값 one hot encoding
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y_one_hot)

# cost = (y_one_hot * -torch.log(hypothesis)).sum(dim = 1).mean()
# print(cost)
# cost = (y_one_hot * - F.log_softmax(z, dim = 1)).sum(dim = 1).mean()
# print(cost)
# cost = F.nll_loss(F.log_softmax(z, dim = 1), y)     # Negative Log Likelihood
# print(cost)
cost = F.cross_entropy(z, y)
print(cost)