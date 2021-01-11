# 데이터
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 70)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 70)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print("x_train.shape: ", x_train.shape)     # (96, 4)
print("x_test.shape: ", x_test.shape)       # (30, 4)
print("x_val.shape: ", x_val.shape)         # (24, 4)
print("y_train.shape: ", y_train.shape)     # (96,)
print(y_train[:5])

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

y  = y.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

enc = OneHotEncoder()
enc.fit(y)
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()
y_val = enc.transform(y_val).toarray()


x_train = x_train.reshape(x_train.shape[0], 2, 2, 1)
x_test = x_test.reshape(x_test.shape[0]   , 2, 2, 1)
x_val = x_val.reshape(x_val.shape[0]      , 2, 2, 1)


# 모델 구성
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Dropout, Flatten

# model = Sequential()
# model.add(Conv2D(filters = 10, kernel_size = 2, strides = 1, padding = 'same', input_shape = (4, 1, 1)))
# model.add(MaxPooling2D(pool_size = (2,1)))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Dense(100))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(3), activation = 'softmax')

input1 = Input(shape = (2, 2, 1))
dense1 = Conv2D(filters = 50, kernel_size = 2, strides = 1, padding = 'same')(input1)
dense1 = MaxPooling2D(pool_size = (2,1))(dense1)
dense1 = Dense(50, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(3, activation = 'sigmoid')(dense1)
model = Model(inputs = input1, outputs = output1)

model.summary()

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
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
print(y_test_predict[:10])

# input_shape = (4, 1, 1)
# R2:  0.9300520490539629
# loss:  0.014479724690318108
# acc:  1.0

# input_shape = (2, 2, 1)
# R2:  0.9066245289660714
# loss:  0.03662088140845299
# acc:  0.9666666388511658