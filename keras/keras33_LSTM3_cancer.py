# 싸이킷런 데이터 셋
# LSTM으로 모델링
# Dense와 성능 비교
# 이진분류


import numpy as np

# 데이터
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)         # (569, 30)
print(np.max(x), np.min(x))     # (569,)
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

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)


# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 = Input(shape = (30, 1))
lstm1 = LSTM(10, activation = "relu")(input1)
dense1 = Dense(100, activation = "relu")(lstm1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(400, activation = "relu")(dense1)
dense1 = Dense(500, activation = "relu")(dense1)
dense1 = Dense(400, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(100, activation = "relu")(dense1)
dense1 = Dense(10, activation = "relu")(dense1)
output1 = Dense(1,  activation = "sigmoid")(dense1)
model = Model(inputs = input1, outputs = output1)


# Compile and Fit and Early_Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 10, mode = "auto")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["acc"])
model.fit(x_train, y_train, epochs = 200, batch_size = 6, validation_data = (x_val, y_val), callbacks = early_stopping)


# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
y_test_predict = model.predict(x_test)
print("y_test_predict: ", y_test_predict.shape)
print("y_test_predict: \n", y_test_predict)
print("y_test: \n", y_test)

loss = model.evaluate(x_test, y_test, batch_size = 6)
rmse = RMSE(y_test, y_test_predict)
R2 = r2_score(y_test, y_test_predict)

print("loss: ", loss)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2)

result = model.predict(x_test)
print("result: \n", result)

# loss:  [0.04091471806168556, 0.09386774897575378]
# RMSE:  0.20227388135198587
# R2_SCORE:  0.8223428863644318


# 아래는 무슨 경우지?
# Epoch 46/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.0783 - mae: 0.1530 - val_loss: 0.0913 - val_mae: 0.1993
# Epoch 47/200
# 61/61 [==============================] - 3s 50ms/step - loss: 0.1062 - mae: 0.1893 - val_loss: 0.1006 - val_mae: 0.1694
# Epoch 48/200
# 61/61 [==============================] - 3s 50ms/step - loss: 0.0808 - mae: 0.1546 - val_loss: 0.0848 - val_mae: 0.1604
# Epoch 49/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.0780 - mae: 0.1390 - val_loss: 0.0952 - val_mae: 0.1579
# Epoch 50/200
# 61/61 [==============================] - 3s 50ms/step - loss: 0.0740 - mae: 0.1366 - val_loss: 0.0784 - val_mae: 0.1382
# Epoch 51/200
# 61/61 [==============================] - 3s 50ms/step - loss: 0.0752 - mae: 0.1307 - val_loss: 0.0869 - val_mae: 0.1421
# Epoch 52/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.0695 - mae: 0.1231 - val_loss: 0.0784 - val_mae: 0.1341

# 여기서부터 과적합 발생! EarlyStopping을 10개정도로 잡자!
# Epoch 53/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.5556 - mae: 0.5621 - val_loss: 0.6703 - val_mae: 0.6703
# Epoch 54/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.6126 - mae: 0.6126 - val_loss: 0.6703 - val_mae: 0.6703
# Epoch 55/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.6126 - mae: 0.6126 - val_loss: 0.6703 - val_mae: 0.6703
# Epoch 56/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.6126 - mae: 0.6126 - val_loss: 0.6703 - val_mae: 0.6703
# Epoch 57/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.6126 - mae: 0.6126 - val_loss: 0.6703 - val_mae: 0.6703
# Epoch 58/200
# 61/61 [==============================] - 3s 50ms/step - loss: 0.6126 - mae: 0.6126 - val_loss: 0.6703 - val_mae: 0.6703
# Epoch 59/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.6126 - mae: 0.6126 - val_loss: 0.6703 - val_mae: 0.6703
# Epoch 60/200
# 61/61 [==============================] - 3s 49ms/step - loss: 0.6126 - mae: 0.6126 - val_loss: 0.6703 - val_mae: 0.6703