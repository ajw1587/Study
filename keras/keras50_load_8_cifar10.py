import numpy as np

x_train = np.load('../data/npy/cifar10_c10_x_train.npy')
x_test = np.load('../data/npy/cifar10_c10_x_test.npy')
y_train = np.load('../data/npy/cifar10_c10_y_train.npy')
y_test = np.load('../data/npy/cifar10_c10_y_test.npy')

print(x_train)
print(y_train)
print(x_train.shape, y_train.shape)

# 모델을 완성하시오.
x_train = x_train/255.
x_test = x_test/255.

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Flatten

input1 = Input(shape = (32, 32, 3))
conv2d = Conv2D(50, 2, strides = 1, padding = 'same', input_shape = (32, 32, 3))(input1)
dense1 = MaxPooling2D()(conv2d)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs = input1, outputs = output1)

model.summary()


# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint', monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 5, batch_size = 32, validation_split = 0.2, callbacks = [es, cp, tb])


# Evaluate and Predict
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)