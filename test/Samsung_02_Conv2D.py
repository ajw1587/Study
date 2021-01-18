import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

# 인덱스 변경       '일자' -> 0,1,2,3,4...
dataset = dataset.reset_index(drop = True)
print(dataset.index)

# npy 형식으로 변경
dataset = dataset.to_numpy()
print(dataset.shape)        # (2398, 8)

# x, x_predict, y 로 나눠주기
size = 5
column = 8
# 함수 생성
def make_x(dataset, size):
    x = []
    for i in range(dataset.shape[0]-size):        # dataset.shape[0]: 2398 2393~2397
        subset = dataset[i:i+size,:]
        x.append(subset)
    return np.array(x)

x = make_x(dataset, size)      # (2393, 5, 8)
x_predict = np.array(dataset[dataset.shape[0]-5:,0:])
x_predict = x_predict.reshape(1, x_predict.shape[0], x_predict.shape[1])
y = dataset[5:, 3]
print(y[-2:])

# train, test, val 나눠주기
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)

# 전처리
x_train = x_train.reshape(x_train.shape[0], size*column)
x_test = x_test.reshape(x_test.shape[0], size*column)
x_val = x_val.reshape(x_val.shape[0], size*column)
x_predict = x_predict.reshape(x_predict.shape[0], size*column)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_predict = scaler.transform(x_predict)

x_train = x_train.reshape(x_train.shape[0], size, column)
x_test = x_test.reshape(x_test.shape[0], size, column)
x_val = x_val.reshape(x_val.shape[0], size, column)
x_predict = x_predict.reshape(x_predict.shape[0], size, column)

file_path = '../data/npy/Samsung_xy_2.npz'
np.savez(file_path, x_train = x_train, x_test = x_test, x_val = x_val, x_predict = x_predict,
                    y_train = y_train, y_test = y_test, y_val = y_val)
#=================================================================================================

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], x_predict.shape[2], 1)

# 모델링
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MaxPooling2D, Conv2D, Flatten

input1 = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
dense1 = Conv2D(filters = 128, kernel_size = (2,2), strides = (1, 1), padding = 'same')(input1)
dense1 = MaxPooling2D(pool_size = 2)(dense1)
dense1 = Dense(256, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

# Compile and Fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
file_path = '../data/modelcheckpoint/Samsung_Conv2D_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'val_loss', patience = 20, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = 'mae')
model.fit(x_train, y_train, batch_size = 16, epochs = 1000, validation_data = (x_val, y_val), callbacks = [es, cp])

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

y_test_predict = model.predict(x_test)
rmse = RMSE(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)

loss, mae = model.evaluate(x_test, y_test, batch_size = 16)
print('loss: ', loss)
print('mae: ', mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", r2)
print("y_predict[-1]: ", y_test_predict[-1])
print("y_test[-1]: ", y_test[-1])

result = model.predict(x_predict)
print('result: ', result)

# loss:  4587603.0
# mae:  1657.71484375
# RMSE:  2141.8689539312827
# R2_SCORE:  0.9734132215446005
# y_predict[-1]:  [24183.002]
# y_test[-1]:  26700.0
# result:  [[87039.016]]