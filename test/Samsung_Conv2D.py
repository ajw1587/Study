import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

dataset= pd.read_csv('./test/삼성전자.csv', engine= 'python', encoding= 'CP949', thousands= ',', index_col= '일자')
dataset_0114 = pd.read_csv('./test/삼성전자2.csv', engine= 'python', encoding= 'CP949', thousands= ',', index_col= '일자')

# 삼성전자.csv 결측치 없애주기
print(dataset.isnull().sum())
print(dataset.shape)        # (2400, 14)
dataset.drop(['2018-05-03', '2018-05-02', '2018-04-30'], inplace = True)
print(dataset.shape)        # (2397, 14)

# 값 /50 해주기
dataset.loc['2018-04-27':, '시가':'종가'] = dataset.loc['2018-04-27':, '시가':'종가']/50

# 삼성전자2.csv 열 맞춰주기
dataset_0114.drop(['전일비', 'Unnamed: 6'], axis = 1, inplace = True)       # axis = 1: 열, 0: 행
print(dataset_0114.shape)   # (60, 14)

# 삼성전자2.csv 필요데이터 추출
dataset_0114 = dataset_0114.loc[['2021-01-14']]
print(dataset_0114.shape)   # (1, 14)

# 삼성전자.csv + 삼성전자2.csv concat
dataset = pd.concat([dataset, dataset_0114])
print(dataset.shape)        # (2398, 14)
print(dataset.index)
print(dataset.columns)

# 삼성전자 정렬
dataset = dataset.sort_index()
print(dataset.iloc[-1, :])

# 데이터 타입 변경  (시가, 고가, 저가, 종가, 거래량, 금액(백만), 신용비, 외인비)
dataset.astype = {
    '시가': np.float,
    '고가': np.float,
    '저가': np.float,
    '종가': np.float,
    '거래량': np.float,
    '금액(백만)': np.float,
    '신용비': np.float,
    '외인비': np.float
}

# 필요 컬럼으로 추리기
dataset = dataset.loc[:, ['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '신용비', '외인비']]
print(dataset.columns)
print(dataset.index)

# 인덱스 변경
dataset = dataset.reset_index(drop = True)
print(dataset.index)

# npy 형식으로 변경
dataset = dataset.to_numpy()
print(dataset.shape)        # (2398, 8)

# x, x_predict, y 로 나눠주기
# 함수 생성
def make_x(dataset, size):
    x = []
    for i in range(dataset.shape[0]-size):        # dataset.shape[0]: 2398 2393~2397
        subset = dataset[i:i+size,:]
        x.append(subset)
    return np.array(x)

x = make_x(dataset, 5)      # (2393, 5, 8)
x_predict = dataset[dataset.shape[0]-1,:]
y = dataset[:, 3]
