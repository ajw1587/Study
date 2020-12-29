import numpy as np
from sklearn.model_selection import train_test_split

x = np.array([range(100), range(301,401), range(1, 101), range(501, 601), range(201, 301)])       #(5,100)
y = np.array([range(711, 811), range(1, 101)]) #(2,100)

x = np.transpose(x)
y = np.transpose(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2)
print("x_train: ", x_train.shape)   #(80,5)
print("y_train: ", y_train.shape)   #(80,2)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(Dense(10, input_dim = 5))       # input_dim = x의 열(컬럼) 갯수
model.add(Dense(10, input_shape = (5,)))
# input_shape는 input_dim이 표현할 수 없는 데이터 표현이 가능하다.
# ex) (500, 100, 3) -> input_shape(100, 3)
#     (10000, 28, 28, 3) -> input_shape(28, 28, 3)

model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))                     # y의 열 갯수

# 컴파일
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train ,y_train, epochs = 100, batch_size = 1, validation_split = 0.2, verbose = 3)      # verbose = 0 -> 훈령과정을 생략해줌

'''
1. verbose = 0 -> 전부 생략

2. verbose = 1 -> Epoch 100/100
                  51/51 [==============================] - 0s 2ms/step - loss: 2.1767e-09 - mae: 3.3053e-05 - val_loss: 1.6217e-09 - val_mae: 3.0297e-05
                  전부 출력

3. verbose = 2 -> Epoch 100/100
                  51/51 - 0s - loss: 2.5012e-09 - mae: 3.7623e-05 - val_loss: 1.7093e-09 - val_mae: 3.1104e-05
                  step별 처리시간 생략

4. verbose = 3 -> Epoch 100/100
                  Epoch제외 전부 생략
'''

# 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print("loss: ", loss)
print("mae: ", mae)

x_predict = np.array([range(1, 6), range(11, 16)]) #(2,5)
y_predict = model.predict(x_test)
print("y_predict: ", "\n", y_predict)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print("RMSE: ", RMSE(y_test, y_predict))
print("R2: ", r2)

x_predict2 = np.array([100, 402, 101, 00, 401])
print("x_predict2: ", x_predict2.shape)
# x_predict2 = np.transpose(x_predict2) -> 변환을 해봤자 1차원인 스칼라이기 때문에 효과 없다.
x_predict2 = x_predict2.reshape(1,5)    # [1, 2, 3, 4, 5] -> [ [1, 2, 3, 4, 5] ]
print("x_predict2: ", x_predict2.shape)

y_predict2 = model.predict(x_predict2)
print("y_predict2: ", y_predict2)