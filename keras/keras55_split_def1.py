import numpy as np

# 리스트 -> 1차원 행열
a = np.array(range(1, 11))      # 1~10
size = 6
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(aaa)
    return np.array(aaa)

# 2차원 행열 -> 3차원 행열
def make_xy(dataset, idx, col):
    x = []
    y = []
    for i in range(dataset.shape[0] - idx - 1):
      x_subset = dataset[i:i+idx, 0:col]
      y_subset = dataset[i+idx : i+idx+2, col-1]
      x.append(x_subset)
      y.append(y_subset)
    return np.array(x), np.array(y)

dataset = split_x(a, size)
x, y = split_xy1(dataset, 2, 3)
print(dataset)
print(len(dataset))
print(x)
print(x.shape)
print(y)
print(y.shape)