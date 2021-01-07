# 텐서플로 데이터 셋
# LSTM으로 모델링
# Dense와 성능 비교


# 1. 데이터
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split = 0.2, seed = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 70)


# 데이터 전처리
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
from tensorflow.keras.layers import Input, Dense, LSTM
input1 = Input(shape = (13, 1))
lstm1 =  LSTM(10, activation = "relu")(input1)
dense1 = Dense(10, activation = "relu")(lstm1)
dense1 = Dense(20, activation = "relu")(dense1)
dense1 = Dense(30, activation = "relu")(dense1)
dense1 = Dense(50, activation = "relu")(dense1)
dense1 = Dense(30, activation = "relu")(dense1)
dense1 = Dense(20, activation = "relu")(dense1)
dense1 = Dense(10, activation = "relu")(dense1)
output1 = Dense(1,  activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)


# Compile and Fit and Early_Stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 30, mode = "auto")
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train, y_train, epochs = 200, batch_size = 10, validation_data = (x_val, y_val), callbacks = early_stopping)


# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
y_test_predict = model.predict(x_test)
print("y_test_predict: ", y_test_predict.shape)
print("y_test_predict: \n", y_test_predict)
print("y_test: \n", y_test)

loss = model.evaluate(x_test, y_test, batch_size = 10)
rmse = RMSE(y_test, y_test_predict)
R2 = r2_score(y_test, y_test_predict)

print("loss: ", loss)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2)

result = model.predict(x_predict)
print("result: \n", result)
