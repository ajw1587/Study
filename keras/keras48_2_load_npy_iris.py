import numpy as np

x_data = np.load('../data/npy/iris_x.npy')
y_data = np.load('../data/npy/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)

# 모델을 완성하시오.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size = 0.8, shuffle = True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = True)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

## 원핫인코딩 OneHotEncoding
# to_categorical
# from tensorflow.keras.utils import to_categorical
# # from keras.utils.np_utils import to_categorical
# y = to_categorical(y)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

# OneHotEncoding
from sklearn.preprocessing import OneHotEncoder
# Endcoding 전에 y의 행렬을 바꿔줘야한다.
y_data = y_data.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_val = y_val.reshape(-1,1)

enc = OneHotEncoder()
enc.fit(y_data)
y_data = enc.transform(y_data).toarray()
y_train = enc.transform(y_train).toarray()
y_test = enc.transform(y_test).toarray()
y_val = enc.transform(y_val).toarray()

print(y_data)
print(y_data.shape)

# 3. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation = "relu", input_shape = (4,)))
model.add(Dense(100, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(3, activation = "softmax"))     # 다중분류의 경우 나누고자 하는 종류의 숫자를 기입하고 softmax를 사용한다.
                                                # 원핫인코딩, to_categorical -> wikidocs.net/22647


# 4. Compile and Train
from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, validation_data = (x_val, y_val), batch_size = 3)
loss, acc = model.evaluate(x_test, y_test)
y_test_predict = model.predict(x_test)
rmse = RMSE(y_test, y_test_predict)

print("loss: ", loss)
print("acc: ", acc)
print("RMSE: ", rmse)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])

# 결과치 나오게 할것 argmax
# print(tf.argmax(y_predict,0).numpy())
# print(tf.argmax(y_predict,1).numpy())

# y_predict
# [[1.3078864e-07 3.1127499e-02 9.6887231e-01]
#  [3.1477865e-03 9.9130064e-01 5.5515943e-03]
#  [9.6905241e-03 6.6673499e-01 3.2357448e-01]
#  [2.5219508e-03 9.9662751e-01 8.5058995e-04]]



print(np.argmax(y_predict, axis = 0))         # 열방향
print(np.argmax(y_predict, axis = 1))         # 행방향
print(tf.argmax(y_predict, 0))                # 열방향
print(tf.argmax(y_predict, 1))                # 행방향