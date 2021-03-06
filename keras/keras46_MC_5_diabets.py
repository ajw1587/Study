# 실습
# 드랍아웃 적용

import numpy as np

# 데이터
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)         # (442, 10)
print(np.max(x), np.min(x))     # (442,)
print(np.max(x[0]), np.min(x[0]))

# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 66)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout

input1 = Input(shape = (10,))
dense1 = Dense(100, activation = "relu")(input1)
dense1 = Dropout(0.2)(dense1)           # 100개중 0.8개 사용
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)           # 200개중 0.8개 사용
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)           # 300개중 0.8개 사용
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)           # 200개중 0.8개 사용
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dropout(0.2)(dense1)           # 100개중 0.8개 사용
output1 = Dense(1,  activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)

# Compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
file_path = '../data/modelcheckpoint/k46_5_diabets_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, batch_size = 1, epochs = 200, validation_data = (x_val, y_val), callbacks = [es, cp])

# 평가 및 예측
loss, mae = model.evaluate(x_test, y_test, batch_size = 1)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("loss: ", loss)
print("mae: ", mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", r2)

# 일반
# loss:  3345.530517578125
# mae:  47.99224853515625
# RMSE:  57.8405595780664
# R2_SCORE:  0.48451317300728847


# x = x/(np.max(x))
# loss:  3579.1552734375
# mae:  49.40424346923828
# RMSE:  59.82604087926021
# R2_SCORE:  0.4328804440523756


# scaler.fit(x)
# loss:  3211.023193359375
# mae:  46.74753189086914
# RMSE:  56.665891041005054
# R2_SCORE:  0.49121120196113477


# scaler.fit(x_train)
# loss:  3182.913330078125
# mae:  46.85000991821289
# RMSE:  56.41731273725618
# R2_SCORE:  0.5095695895801424

# Dropout 적용
# loss:  5576.54345703125
# mae:  58.680755615234375
# RMSE:  74.67625692708027
# R2_SCORE:  0.1407536770417468