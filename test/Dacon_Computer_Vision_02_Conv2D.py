# Computer_Vision
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
'''
# 1. Train 데이터 y -> 0~9, 다중분류
file_path = '../data/csv/Computer_Vision/data/train.csv'
dataset = pd.read_csv(file_path, engine = 'python', encoding = 'CP949', index_col = 0)
# print(type(dataset))        # <class 'numpy.ndarray'>
# print(dataset.shape)        # (2048, 786)

x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
# print(type(x))              # <class 'numpy.ndarray'>
# print(x.shape)              # (2048, 785)
# print(type(y))              # <class 'numpy.ndarray'>
# print(y.shape)              # (2048,)


# 2. PCA 데이터가 문자라서 에러나 난다. ASCII로 바꿔줘야 하나 아니면 그냥 버려야하나 모르겠네

# 2-1. 문자 데이터 버리기
x = x[:, 1:].astype(np.float)
x = x/255.

# 3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)
print(x_train.shape)                              # (1310, 784)
print(x_val.shape)                                # (328, 784)
print(x_test.shape)                               # (410, 784)


# # 784 =  28 * 28 * 1
shape1 = 28
shape2 = 28
shape3 = 1
x_train = x_train.reshape(x_train.shape[0], shape1, shape2, shape3)
x_val = x_val.reshape(x_val.shape[0], shape1, shape2, shape3)
x_test = x_test.reshape(x_test.shape[0], shape1, shape2, shape3)

# 4. OneHotEncoding
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 5. 모델
input1 = Input(shape = (shape1, shape2, shape3))
conv2d = Conv2D(256, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(input1)
dense1 = Conv2D(128, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(conv2d)
dense1 = MaxPooling2D(pool_size = (2, 2))(dense1)
dense1 = Conv2D(256, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
dense1 = Conv2D(128, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
dense1 = Conv2D(64, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(dense1)
dense1 = MaxPooling2D(pool_size = (2, 2))(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Flatten()(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
output1 = Dense(10, activation = 'softmax')(dense1)
model = Model(inputs = input1, outputs = output1)

# 5. Fit
cp_path = '../data/modelcheckpoint/Computer_Vision/Computer_Vision3_{epoch: 03d}_{val_loss: .4f}.hdf5'
cp = ModelCheckpoint(cp_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
opti = Adam(learning_rate = 0.0001)
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, mode = 'auto')
model.compile(loss = 'categorical_crossentropy', optimizer = opti, metrics = ['acc'])
model.fit(x_train, y_train, epochs = 1000, batch_size = 32, validation_data = (x_val, y_val), 
          callbacks = [es, reduce_lr, cp])

# 6. Evaluate, Pred
loss, acc = model.evaluate(x_test, y_test)
y_pred = model.predict(x_test)
print('loss: ', loss)
print('acc: ', acc)
# loss:  1.8897967338562012
# acc:  0.7219512462615967
'''

# Predict ########################################################################
test_file_path = '../data/csv/Computer_Vision/data/test.csv'
submission_path = '../data/csv/Computer_Vision/data/submission.csv'
model_path = '../data/modelcheckpoint/Computer_Vision/Computer_Vision3_ 24_ 0.9731.hdf5'
test_dataset = pd.read_csv(test_file_path, engine = 'python', encoding = 'CP949', index_col = 0)

# 1. 데이터
print(test_dataset.shape)           # (20480, 784)
real_x_test = test_dataset.values
real_x_test = real_x_test[:, 1:].astype(np.float)
real_x_test = real_x_test/255.
real_x_test = real_x_test.reshape(real_x_test.shape[0], 28, 28, 1)

# 2. 모델
model = load_model(model_path)

# 3. Predict
y_predict = model.predict(real_x_test)
y_predict = np.argmax(y_predict)

# 4. submission
submission_data = pd.read_csv(submission_path, encoding = 'CP949', engine = 'python')

submission_data['digit'] = np.argmax(model.predict(real_x_test), axis = 1)
print(submission_data.head())

submission_data.to_csv('../data/modelcheckpoint/Computer_Vision/submission_data3.csv', index = False)
