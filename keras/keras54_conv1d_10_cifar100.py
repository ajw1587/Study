import numpy as np
from tensorflow.keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print("x_train.shape: ", x_train.shape)       # (50000, 32, 32, 3)
print("y_train.shape: ", y_train.shape)       # (50000, 1)
print("x_test.shape: ", x_test.shape)         # (10000, 32, 32, 3)
print("y_test.shape: ", y_test.shape)         # (10000, 1)

print("x_train[0].shape", x_train[0].shape)   # (32, 32, 3)
print("y_train[0].shape", y_train[0].shape)   # (1,)

x_train = x_train.reshape(50000, 1024, 3)
x_test = x_test.reshape(10000, 1024, 3)
x_train = x_train/255.
x_test = x_test/255.

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Dropout, MaxPooling1D, Flatten

input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
conv1d = Conv1D(50, 2, strides = 1, padding = 'same')(input1)
dense1 = MaxPooling1D()(conv1d)
dense1 = Conv1D(50, 2, strides = 1, padding = 'same')(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(100, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()(dense1)
output1 = Dense(100, activation = 'softmax')(dense1)
model = Model(inputs = input1, outputs = output1)

model.summary()


# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
file_path = '../data/modelcheckpoint/k46_3_cifar100_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 3, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 50, batch_size = 100, validation_split = 0.2, callbacks = [es])


# Evaluate and Predict
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)
y_predict = model.predict(x_test)

print("loss: ", loss)
print("acc: ", acc)

# Conv1D 적용 - 성능 매우 안좋음
# loss:  7.918727397918701
# acc:  0.24650000035762787