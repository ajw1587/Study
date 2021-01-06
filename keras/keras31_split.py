import numpy as np

a = np.array(range(1, 11))      # 1~10
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        # aaa.append([item for item in subset])
        aaa.append(subset)
    print(aaa)
    return np.array(aaa)

dataset = split_x(a, size)
print("===========================")
print(dataset)

# x = np.array(range(5))
# print(len(x))
# print(x)
# print(x[0:5])