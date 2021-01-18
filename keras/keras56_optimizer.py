# learning rate
import numpy as np

# 1. 데이터
x = np.array([1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10])
y = np.array([1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim = 1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# 3. Compile, Fit
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

# optimizer = Adam(lr = 0.1)
# loss:  6.3298821449279785 결과물:  [[14.363474]]
# optimizer = Adam(lr = 0.01)
# loss:  1.131184013646036e-12 결과물:  [[10.999998]]
# optimizer = Adam(lr = 0.001)
# loss:  0.002553344937041402 결과물:  [[11.038349]]
# optimizer = Adam(lr = 0.0001)
# loss:  9.373296052217484e-06 결과물:  [[10.995134]]

# optimizer = Adadelta(lr = 0.1)
# loss:  0.004442903213202953 결과물:  [[11.13461]]
# optimizer = Adadelta(lr = 0.01)
# loss:  0.0003502063627820462 결과물:  [[11.025873]]
# optimizer = Adadelta(lr = 0.001)
# loss:  8.72844123840332 결과물:  [[5.696331]]
# optimizer = Adadelta(lr = 0.0001)
# loss:  40.84844207763672 결과물:  [[-0.33763584]]

# optimizer = Adamax(lr = 0.1)
# loss:  1.5442473966231773e-07 결과물:  [[10.999311]]
# optimizer = Adamax(lr = 0.01)
# loss:  2.9594106028610345e-13 결과물:  [[11.]]
# optimizer = Adamax(lr = 0.001)
# loss:  1.750065692363023e-08 결과물:  [[10.99986]]
# optimizer = Adamax(lr = 0.0001)
# loss:  0.003658442525193095 결과물:  [[10.929069]]

# optimizer = Adagrad(lr = 0.1)
# loss:  0.07544072717428207 결과물:  [[10.457975]]
# optimizer = Adagrad(lr = 0.01)
# loss:  6.707874717903906e-07 결과물:  [[10.9993515]]
# optimizer = Adagrad(lr = 0.001)
# optimizer = Adagrad(lr = 0.0001)
# loss:  0.005851424299180508 결과물:  [[10.904166]]

# optimizer = RMSprop(lr = 0.1)
# loss:  91803696.0 결과물:  [[-20653.424]]
# optimizer = RMSprop(lr = 0.01)
# loss:  4.317753314971924 결과물:  [[6.51118]]
# optimizer = RMSprop(lr = 0.001)
# loss:  0.005148896481841803 결과물:  [[11.155007]]
# optimizer = RMSprop(lr = 0.0001)
# loss:  0.006591530982404947 결과물:  [[11.153156]]

# optimizer = SGD(lr = 0.1)
# loss:  nan 결과물:  [[nan]]
# optimizer = SGD(lr = 0.01)
# loss:  nan 결과물:  [[nan]]
# optimizer = SGD(lr = 0.001)
# loss:  3.9958885622581874e-07 결과물:  [[10.999952]]
# optimizer = SGD(lr = 0.0001)
# loss:  0.001081277965568006 결과물:  [[10.965037]]

# optimizer = Nadam(lr = 0.1)
# loss:  14628.302734375 결과물:  [[272.21127]]
# optimizer = Nadam(lr = 0.01)
# loss:  1.2671819323017974e-11 결과물:  [[11.000004]]
# optimizer = Nadam(lr = 0.001)
# loss:  5.3353967814473435e-05 결과물:  [[10.990157]]
# optimizer = Nadam(lr = 0.1)
# loss:  1.0031792641029824e-07 결과물:  [[11.000315]]

model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])
model.fit(x, y, epochs = 100, batch_size = 1)

# 4. Evaluate, Predict
loss, mse = model.evaluate(x, y, batch_size = 1)
y_predict = model.predict([11])
print('loss: ', loss, '결과물: ', y_predict)