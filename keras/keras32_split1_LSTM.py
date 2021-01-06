import numpy as np

a = np.array(range(1, 11))      # 1,10

def split_x(a, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i: (i+size)]
        aaa.append(subset)
    print(aaa)
    return np.array(aaa)

dataset = split_x(a, 5)
print(dataset)
print(dataset.shape)

x = dataset[:, :4]
y = dataset[:, 4:]

x = x.reshape(6,4,1)

print(x.shape)
print(y.shape)


# 모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

input1 = Input(shape = (4,1))
dense1 = LSTM(10)(input1)
dense1 = Dense(20)(dense1)
dense1 = Dense(50)(dense1)
dense1 = Dense(20)(dense1)
dense1 = Dense(20)(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs = input1, outputs = output1)

# Compile and Fit
model.compile(loss = "mse", optimizer = "adam")
model.fit(x, y, epochs = 150, batch_size = 1)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss: ", loss)

x_predict = np.array([7, 8, 9, 10])
x_predict = x_predict.reshape(1, 4, 1)

result = model.predict(x_predict)
print("result: ", result)