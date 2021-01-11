# 데이터
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print("x_train.shape: ", x_train.shape)     # (282, 10)
print("x_test.shape: ", x_test.shape)       # (89, 10)
print("x_val.shape: ", x_val.shape)         # (71, 10)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1, 1)


# 모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Dropout, Flatten

# model = Sequential()
# model.add(Conv2D(filters = 10, kernel_size = 2, strides = 1, padding = 'same', input_shape = (10, 1, 1)))
# model.add(MaxPooling2D(pool_size = (2,1)))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(1))

input1 = Input(shape = (10, 1, 1))
dense1 = Conv2D(filters = 50, kernel_size = 2, strides = 1, padding = 'same')(input1)
dense1 = MaxPooling2D(pool_size = (2,1))(dense1)
dense1 = Dense(50, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.1)(dense1)
dense1 = Dense(50, activation = 'relu')(dense1)
dense1 = Dropout(0.1)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

model.summary()

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'mse', optimizer = 'adam', metrics = 'mae')
model.fit(x_train, y_train, batch_size = 1, epochs = 200, validation_data = (x_val, y_val), callbacks = es)

# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test, y_test_predict):
  return np.sqrt(mean_squared_error(y_test, y_test_predict))

loss, mae = model.evaluate(x_test, y_test)
y_test_predict = model.predict(x_test)

print("R2: ", r2_score(y_test, y_test_predict))
print("loss: ", loss)
print("mae: ", mae)

# R2:  0.4269673244560289
# loss:  3632.939697265625
# mae:  49.5133857727050

# R2:  0.4454931033436883
# loss:  3515.489013671875
# mae:  48.997920989990234