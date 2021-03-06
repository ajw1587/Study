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


# 모델구성 아래의 Sequential 모델과 Function 모델의 구성은 같다
from tensorflow.keras.models import Sequential, Model      # Sequenrial은 순차적인 모델
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(5,))
dense1 = Dense(5, activation = "relu")(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs = input1, outputs = outputs)
model.summary()                                             # 모델 구성 확인

# model = Sequential()
# model.add(Dense(5, activation = "relu", input_shape = (5,)))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(2))
# model.summary()



# 컴파일
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train ,y_train, epochs = 5000, batch_size = 1, validation_split = 0.2, verbose = 3)      # verbose = 0 -> 훈령과정을 생략해줌

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