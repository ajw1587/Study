import numpy as np
from sklearn.datasets import load_diabetes

# 데이터
dataset = load_diabetes()
x = dataset.data        #(442, 10)
y = dataset.target      #(442,)
# print(x.shape)
# print(y.shape)
# print(dataset.feature_names)
# print(dataset.DESCR)

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
from tensorflow.keras.layers import Dense, Input
input1 = Input(shape = (10,))
dense1 = Dense(30, activation = "relu")(input1)
dense1 = Dense(30, activation = "relu")(dense1)
dense1 = Dense(20, activation = "relu")(dense1)
dense1 = Dense(10, activation = "relu")(dense1)
dense1 = Dense(5,  activation = "relu")(dense1)
output1 = Dense(1, activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
early_stopping = EarlyStopping(monitor = "loss", mode = "auto", patience = 30)
model.fit(x_train, y_train, batch_size = 6, epochs = 200, validation_data = (x_val, y_val), callbacks = early_stopping)

# 평가 및 예측
loss, mae = model.evaluate(x_test, y_test, batch_size = 6)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("loss(mse): ", loss)
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

# early_stopping 30
# loss(mse):  3186.040771484375
# mae:  46.55258560180664
# RMSE:  56.44502396349416
# R2_SCORE:  0.5090876890750897