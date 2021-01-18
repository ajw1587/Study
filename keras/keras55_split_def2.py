import numpy as np

# 리스트 -> 1차원 행열
a = np.array(range(1, 16))      # 1~10
size = 6
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    return np.array(aaa)

# 2차원 행열 -> 3차원 행열
def split_xy2(dataset, x_low, x_col, y_low, y_col):
    x, y = [], []
    for i in range(len(dataset) - x_low - y_low ):
        x_subset = dataset[i : i + x_low, 0 : x_col]
        x.append(x_subset)

    for i in range(len(dataset) - y_low -1):
        if (i + x_low + y_low) > len(dataset):
            break
        y_subset = dataset[i + x_low : i + x_low + y_low , -y_col:]
        y.append(y_subset)
    x = np.array(x)
    y = np.array(y)
    return x, y


dataset = split_x(a, size)
print(len(dataset))
print(dataset.shape)
x, y = split_xy2(dataset, 5, 3, 2, 4)
print(dataset)
print(x)
print(x.shape)
print(y)
print(y.shape)