import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv('./test/삼성전자.csv', engine = 'python', encoding = 'CP949', thousands = ',')
# 한글 깨짐현상 해결: engine = 'python', encoding = 'CP949'
# 쉼표 없애기: thousands = ','



# 1. 데이터 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# dataset.corr() 상관계수 히트맵으로 상관성 분석
print(dataset.corr())
#===================================================================================================================================================
#            시가      고가      저가      종가      등락률    거래량    금액(백만)  신용비       개인     기관     외인(수량)  외국계    프로그램   외인비
# 종가      0.999706  0.999872  0.999885  1.000000  0.012075 -0.687427 -0.292825 -0.224739 -0.062605  0.058436  0.023520  0.035020  0.085272 -0.576425
#===================================================================================================================================================
# 0.2이상 -0.2이하 선택
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
print(type(dataset.iloc[0, 1]))     # 변환 유무 확인 = <class 'numpy.float64'>

# 2018-04-30 이전 데이터(시가,고가,저가,종가) 50으로 나눠주기
dataset.iloc[665:, 1:5] = dataset.iloc[665:, 1:5]/50


# 결측값 제거 split, concat
# print(dataset.iloc[662,:])      # 2018-05-03
# print(dataset.iloc[664,:])      # 2018-04-30
dataset1 = dataset.iloc[:662, :]
dataset2 = dataset.iloc[665:, :]
print(dataset1.iloc[661])
print(dataset2.iloc[0])
dataset = pd.concat([dataset1, dataset2], ignore_index=True)

# 데이터 정렬
dataset = dataset.sort_values(by = '일자', ascending = True)

# 필요 Column 추출
dataset = dataset.loc[:, ['시가', '고가', '저가', '종가', '거래량', '금액(백만)', '신용비', '외인비']]
print(dataset.iloc[700])

# 최종 데이터 npy형식으로 바꿔주기
np_dataset = dataset.to_numpy()
print(np_dataset.shape)         # (2397, 8)


# MinMaxScaler



# np_dataset에서 x.shape = (2392, 5, 8), y.shape = (2392,) 생성
# x값 (???, 5, 8)
x = []
y = []
for i in range(2392):
    c = []
    for j in range(5):
        b = np_dataset[i+j, :]
        c.append(b)
    x.append(c)
    y.append(np_dataset[i+5,3])

x = np.array(x)
print(x[0])
print(x.shape)      # (2392, 5, 8)

# y값
# y = [row[3] for row in x]      # 3열에 있는 종가를 y값으로
y = np.array(y)
print(y.shape)      # (2392,)
print(y[0])

# npy 저장
np.save('../data/npy/Samsung_x.npy', arr = x)
np.save('../data/npy/Samsung_y.npy', arr = y)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
x = x.reshape(x.shape[0], 40)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
print(x_train.shape)
print(x_test.shape)
print(x_val.shape)

x_train = x_train.reshape(x_train.shape[0], 5, 8)
x_test = x_test.reshape(x_test.shape[0], 5, 8)
x_val = x_val.reshape(x_val.shape[0], 5, 8)

# 2. 모델 구성+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, LSTM, Dense

input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
dense1 = LSTM(100)(input1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(200, activation = 'relu')(dense1)
dense1 = Dense(150, activation = 'relu')(dense1)
dense1 = Dense(150, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)
model.summary()

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
file_path = "../data/modelcheckpoint/Samsung_{epoch:02d}_{val_loss:.4f}.hdf5"
es = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 400, batch_size = 3, validation_data = (x_val, y_val), callbacks = [es, cp])

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

y_predict = model.predict(x_test)
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

loss, mae = model.evaluate(x_test, y_test, batch_size = 3)
print('loss: ', loss)
print('mae: ', mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", r2)