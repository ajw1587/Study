import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv('./test/삼성전자.csv', engine = 'python', encoding = 'CP949', thousands = ',')
dataset_append = pd.read_csv('./test/삼성전자2.csv', engine = 'python', encoding = 'CP949', thousands = ',')

# 데이터 Merge
# 삼성전자2 -> 필요없는 열 제거
print(dataset_append.shape)     # (60, 17)
dataset_append.drop(['전일비', 'Unnamed: 6'], axis = 'columns', inplace = True)
print(dataset_append.shape)     # (60, 15)

# 삼성전자2 -> 필요한 부분 자르기
dataset_append = dataset_append.iloc[0:1, :]
print(dataset_append)
print(dataset_append.shape)     # (1, 15)

# 삼성전자1 + 삼성전자2
dataset = pd.concat([dataset_append, dataset], ignore_index = True)
# print(dataset.shape)          # (2401, 15)


# dataest_append = pd.read_csv('./test/삼성전자2')
# 한글 깨짐현상 해결: engine = 'python', encoding = 'CP949'
# 쉼표 없애기: thousands = ','

# 1. 데이터 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# dataset.corr() 상관계수 히트맵으로 상관성 분석
# print(dataset.corr())
#===================================================================================================================================================
#            시가      고가      저가      종가      등락률    거래량    금액(백만)  신용비       개인     기관     외인(수량)  외국계    프로그램   외인비
# 종가      0.999706  0.999872  0.999885  1.000000  0.012075 -0.687427 -0.292825 -0.224739 -0.062605  0.058436  0.023520  0.035020  0.085272 -0.576425
#===================================================================================================================================================
# 시가, 고가, 저가, 종가, 거래량, 금액(백만), 신용비, 외인비
# 총 8가지 요인

# 자료형 변환(str -> float) -> 시가,고가,저가 50으로 나누기 -> 결측값 제거 -> 일자 오름차준 정렬 -> 필요 Column추출 -> Numpy형식으로 바꾸기
# 데이터 자료형 변환 str -> float
dataset = dataset.astype(
    {'시가': np.float,                # 1열 
     '고가': np.float,                # 2열
     '저가': np.float,                # 3열
     '종가': np.float,                # 4열
     '거래량': np.float,              # 6열
     '금액(백만)': np.float,          # 7열
     '신용비': np.float,              # 9열
     '외인비': np.float}              # 10열
)
# print(type(dataset.iloc[0, 1]))     # 변환 유무 확인 = <class 'numpy.float64'>

# 2018-04-30 이전 데이터(시가,고가,저가,종가) 50으로 나눠주기
dataset.iloc[665:, 1:5] = dataset.iloc[665:, 1:5]/50
# print(dataset.iloc[665:670, 1:5])


# 결측값 제거 split, concat
# print(dataset.iloc[663,:])      # 2018-05-03
# print(dataset.iloc[665,:])      # 2018-04-30
dataset1 = dataset.iloc[:663, :]
dataset2 = dataset.iloc[666:, :]
dataset = pd.concat([dataset1, dataset2], ignore_index=True)

# 데이터 정렬
dataset = dataset.sort_values(by = '일자', ascending = True)

# 필요 Column 추출
dataset = dataset.loc[:, ['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '신용비', '외인비']]

# 최종 데이터 npy형식으로 바꿔주기
np_dataset = dataset.to_numpy()
print(np_dataset.shape)         # (2397, 8)
print(np_dataset[-1, :])

# x값 (???, 5, 8) 5일 단위로 끊기
size = 5
col = 8
x = []
x_predict = []
y = []

for i in range(np_dataset.shape[0] - size): # 2393
    subset = np_dataset[i:i+size, :]
    x.append(subset)
    y.append(np_dataset[i+size, 3])     # i+size, 3

subset = np_dataset[np_dataset.shape[0] - size:,:]
x_predict.append(subset)

# x값
x = np.array(x)
print(x[-1])
print(x.shape)      # (2393, 5, 8)

# x_predict 값
x_predict = np.array(x_predict)
print(x_predict[-1])
print(x_predict.shape)  # (1, 5, 8)

# y값
y = np.array(y)
print(y[2392])
print(y.shape)      # (2393,)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# , shuffle = True, random_state = 100
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)
print(x_train.shape)

x_train = x_train.reshape(x_train.shape[0], 40)
x_test = x_test.reshape(x_test.shape[0], 40)
x_val = x_val.reshape(x_val.shape[0], 40)
x_predict = x_predict.reshape(x_predict.shape[0], 40)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)
print(x_predict.shape)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_predict = scaler.transform(x_predict)

x_train = x_train.reshape(x_train.shape[0], 5, 8)
x_test = x_test.reshape(x_test.shape[0], 5, 8)
x_val = x_val.reshape(x_val.shape[0], 5, 8)
x_predict = x_predict.reshape(x_predict.shape[0], 5, 8)

# npy 저장
np.savez('../data/npy/Samsung_xy.npz', x_train = x_train, x_test = x_test, x_val = x_val, 
                                       y_train = y_train, y_test = y_test, y_val = y_val, x_predict = x_predict)
# 2. 모델 구성+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, LSTM, Dense

input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
dense1 = LSTM(512, activation = 'relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(256, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
dense1 = Dense(8, activation = 'relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)
model.summary()


# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
file_path = "../data/modelcheckpoint/Samsung_{epoch:02d}_{val_loss:f}.hdf5"
es = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 500, batch_size = 16, validation_data = (x_val, y_val), callbacks = [es, cp])


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

# loss:  4649753.5
# mae:  1663.8216552734375
# RMSE:  2156.3287922635222
# R2_SCORE:  0.9744201454072997
# y_predict[-1]:  [24207.791]
# y_test[-1]:  25660.0
# result:  [[90029.24]]

# loss:  6505567.5
# mae:  2019.037353515625
# RMSE:  2550.601147009399
# R2_SCORE:  0.9642106995935816
# y_predict[-1]:  [23850.684]
# y_test[-1]:  25660.0
# result:  [[89362.92]]