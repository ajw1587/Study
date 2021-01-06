# 과제 및 실습  LSTM
# EarlyStopping, 전처리 등등 지금까지 배운 내용 다 넣기
# 데이터는 1~ 100 / x_size: 5
#     x               y
# 1,2,3,4,5           6
# ...
# 95,96,97,98,99    100
# predict 만들것
# 96,97,98,99,100 -> 101
# ...
# 100,101,102,103,104 -> 105
# 예상 predict는 (101, 102, 103, 104, 105)

# 1. 데이터
import numpy as np

sample_data = np.array(range(1,101))

def data_split(data, size):
    arr = []
    for i in range(len(data) - size + 1):
        subset = data[i : (i+size)]
        arr.append(subset)
    return np.array(arr)

dataset = data_split(sample_data, 6)
print(dataset)

x = dataset[:, :5]
y = dataset[:, 5:]
print("x.shape: ", x.shape)     # (95,5)
print("y.shape: ", y.shape)     # (95,1)

# 데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

