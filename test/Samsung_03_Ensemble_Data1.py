# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
filename1 = '../data/csv/삼성전자.csv'
filename2 = '../data/csv/삼성전자0115.csv'

dataset1 = pd.read_csv(filename1, engine = 'python', encoding = 'CP949', thousands = ',', index_col = 0)
dataset2 = pd.read_csv(filename2, engine = 'python', encoding = 'CP949', thousands = ',', index_col = 0)

idx = 5
col = 8
# 함수 생성
def make_xy(dataset, idx):
  x = []
  y = []
  for i in range(dataset.shape[0] - idx - 1):
    x_subset = dataset[i:i+idx, :]
    y_subset = dataset[i+idx : i+idx+2, 0]
    x.append(x_subset)
    y.append(y_subset)
  return np.array(x), np.array(y)

# dataset1 결측치 제거
dataset1 = dataset1.dropna(axis = 0) # (2400, 14) -> (2397, 14)

# 필요 컬럼 추출
dataset1 = dataset1.loc[:, ['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '신용비', '외인비']]  # (2397, 8)
dataset2 = dataset2.loc[:, ['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '신용비', '외인비']]  # (80, 8)

# dataset2 필요 데이터 추출 -> 2021-01-14, 2021-01-15
dataset2 = dataset2.loc[['2021/01/14', '2021/01/15']]

# dataset2 index 이름 바꿔주기
dataset2 = dataset2.rename(index = {'2021/01/14': '2021-01-14',
                                    '2021/01/15': '2021-01-15'})

# string -> float 변환
dataset1 = dataset1.astype(float)
dataset2 = dataset2.astype(float)

# dataset1 값 /50 해주기
dataset1.loc['2018-04-27':, '시가':'종가'] = dataset1.loc['2018-04-27':, '시가':'종가']/50

# dataset1, dataset2 Merge
dataset = pd.concat([dataset1, dataset2])   # (2399, 8)

# dataset 정렬
dataset = dataset.sort_index(ascending = True)

# 코닥스와 행 맞춰주기    2016/08/10 ~ 2021/01/15
dataset = dataset.loc['2016-08-10' : '2021-01-15', :]                       # (1085, 8)

# 인덱스 삭제
dataset = dataset.reset_index(drop = True)

# npy 변환
dataset = dataset.to_numpy()

# train,test, val, predict 데이터 생성
x, y = make_xy(dataset, idx)                                                # (1079, 5, 8)), (1079, 2))
print(x.shape)
print(y.shape)
x_predict = np.array(dataset[-5:, 0:])
x_predict = x_predict.reshape(1, x_predict.shape[0], x_predict.shape[1])    # (1, 5, 8)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False)

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

filepath = '../data/npy/Samsung_StockData.npz'
np.savez(filepath, x_train = x_train, x_test = x_test, x_val = x_val, x_predict = x_predict,
                   y_train = y_train, y_test = y_test, y_val = y_val)