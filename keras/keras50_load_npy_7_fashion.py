import numpy as np

x_train = np.load('../data/npy/fashion_mnist_m_x_train.npy')
x_test = np.load('../data/npy/fashion_mnist_m_x_test.npy')
y_train = np.load('../data/npy/fashion_mnist_m_y_train.npy')
y_test = np.load('../data/npy/fashion_mnist_m_y_test.npy')

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape)

# 모델을 완성하시오.
x_train = x_train/255.
x_test = x_test/255.

# OneHotEncoding
# 여러분이 하시오!
# from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, LSTM

model = Sequential()
model.add(LSTM(10, input_shape = (28, 28)))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

model.summary()

# 실습!! 완성하시오!!!
# 지표는 acc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint', monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 20, epochs = 100, validation_split = 0.2, callbacks = [es, cp, tb])

# 응용
# y_test 10개와 y_test 10개를 출력하시오.
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)