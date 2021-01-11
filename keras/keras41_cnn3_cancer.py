# 데이터
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print("x_train.shape: ", x_train.shape)     # (364, 30)
print("x_test.shape: ", x_test.shape)       # (114, 30)
print("x_val.shape: ", x_val.shape)         # (91, 30)

x_train = x_train.reshape(x_train.shape[0], 15, 2, 1)
x_test = x_test.reshape(x_test.shape[0]   , 15, 2, 1)
x_val = x_val.reshape(x_val.shape[0]      , 15, 2, 1)


# 모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Dropout, Flatten

# model = Sequential()
# model.add(Conv2D(filters = 10, kernel_size = 2, strides = 1, padding = 'same', input_shape = (15, 2, 1)))
# model.add(MaxPooling2D(pool_size = (2,1)))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(1))

input1 = Input(shape = (15, 2, 1))
dense1 = Conv2D(filters = 50, kernel_size = 2, strides = 1, padding = 'same')(input1)
dense1 = MaxPooling2D(pool_size = (2,1))(dense1)
dense1 = Dense(50, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(1, activation = 'sigmoid')(dense1)
model = Model(inputs = input1, outputs = output1)

model.summary()

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 5, epochs = 200, validation_data = (x_val, y_val), callbacks = es)

# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error
def RMSE(y_test, y_test_predict):
  return np.sqrt(mean_squared_error(y_test, y_test_predict))

loss, acc = model.evaluate(x_test, y_test)
y_test_predict = model.predict(x_test)

print("R2: ", r2_score(y_test, y_test_predict))
print("loss: ", loss)
print("acc: ", acc)

# input_shape = (30, 1, 1)
# R2:  0.866502360607436
# loss:  0.11539433896541595
# acc:  0.9561403393745422

# input_shape = (5, 3, 2)
# R2:  0.8537966544232856
# loss:  0.16307231783866882
# acc:  0.9561403393745422

# input_shape = (15, 2, 1)
# R2:  0.856342863745839
# loss:  0.12134335190057755
# acc:  0.9561403393745422