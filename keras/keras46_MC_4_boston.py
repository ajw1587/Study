import numpy as np
from sklearn.datasets import load_boston

# 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)      # (506, 13)
print(y.shape)      # (506,)
print("=========================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))
print(dataset.feature_names)
# print(dataset.DESCR)

# 데이터 전처리 (MinMax)
# x = x/711.              # 맨 뒤에 .을 붙이는 이유는 실수형으로 변환해주기 위해 사용
                        # (x-최소)/(최대-최소) 최솟값이 0이 아닐 경우 0~1 사이의 수로 바꿔주기 위한 연산 수행
print(np.max(x[0]))

# MinMax_Scalar -> 각 Column마다 최소, 최대값을 몰라도 MinMaxScaler를 사용하면 열(Column)마다 자동으로 0 ~ 1 사이로 만들어준다.

# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 66)
print(x_train.shape)
print(y_train.shape)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_shape = (13,), activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(500, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(10,  activation = "relu"))
model.add(Dense(1))

# compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
file_path = '../data/modelCheckpoint/k46_4_boston_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, batch_size = 3, epochs = 200, validation_data = (x_val, y_val), callbacks = [es, cp])

# 평가 및 예측
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
y_predict = model.predict(x_test)
rmse = RMSE(y_test, y_predict)
R2_SCORE = r2_score(y_test, y_predict)
loss, mae = model.evaluate(x_test, y_test, batch_size = 3)

print("loss(mse): ", loss)
print("mae: ", mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2_SCORE)