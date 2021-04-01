import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
filename = '../data/csv/KODEX 코스닥150 선물인버스.csv'

dataset = pd.read_csv(filename, engine = 'python', encoding = 'CP949', thousands = ',', index_col = 0)

idx = 5
col = 8
# 함수 생성
def make_x(dataset, idx):
  x = []
  for i in range(dataset.shape[0] - idx - 1):
    x_subset = dataset[i:i+idx, :]
    x.append(x_subset)
  return np.array(x)

# 2018/04/30, 2018/05/02, 2018/05/03 삭제
dataset = dataset.drop(index = ['2018/04/30', '2018/05/02', '2018/05/03'])    # (1088, 16) -> (1085, 16)
print(dataset.index)                                                          # 2016/08/10 ~ 2021/01/15

# 필요 컬럼 추출
dataset = dataset.loc[:, ['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '신용비', '외인비']]  # (1085, 8)

# string -> float
datset = dataset.astype(float)

# 정렬
dataset = dataset.sort_index(ascending = True)
print(dataset)
# index 삭제
dataset = dataset.reset_index(drop = True)

# npy 변환
dataset = dataset.to_numpy()

# train, test, val, predict 데이터 생성
x = make_x(dataset, idx)                # (1079, 5, 8)
x_predict = dataset[-5:, 0:]            # (5, 8)   
x_predict = x_predict.reshape(1, x_predict.shape[0], x_predict.shape[1])    # (1, 5, 8)
x_train, x_test = train_test_split(x, train_size = 0.8, shuffle = False)
x_train, x_val = train_test_split(x_train, train_size = 0.8, shuffle = False)

# MinMaxScaler
x_train = x_train.reshape(x_train.shape[0], idx*col)
x_test = x_test.reshape(x_test.shape[0], idx*col)
x_val = x_val.reshape(x_val.shape[0], idx*col)
x_predict = x_predict.reshape(x_predict.shape[0], idx*col)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_predict = scaler.transform(x_predict)

x_train = x_train.reshape(x_train.shape[0], idx, col)
x_test = x_test.reshape(x_test.shape[0], idx, col)
x_val = x_val.reshape(x_val.shape[0], idx, col)
x_predict = x_predict.reshape(x_predict.shape[0], idx, col)

filepath = '../data/npy/Samsung_KODEX.npz'
np.savez(filepath, x_train = x_train, x_test = x_test, x_val = x_val, x_predict = x_predict)