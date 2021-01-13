# keras23_3

import numpy as np
# 1. 데이터
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
              [9, 10, 11], [10, 11, 12], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])
x_pred = np.array([50, 60, 70])

# 코딩하시오!!! Conv1D
# 원하는 답은 80
print("x.shape: ", x.shape)     # (13, 3)
print("y.shape: ", y.shape)     # (13,)
x = x.reshape(13, 3, 1)
print(x)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten

model = Sequential()
model.add(Conv1D(filters = 50, kernel_size = 2, strides = 1,
                 padding = 'same', input_shape = (3, 1)))
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(50, 2, 1, padding = 'same'))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Dense(20))
model.add(Dropout(0.2))
model.add(Flatten())            # R2_SCORE을 사용한다면 Flatten은 필수!
model.add(Dense(1))
model.summary()

# Compile and Fit
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
file_path = '../data/modelcheckpoint/k54_conv1d_lstm_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = "mse", optimizer = "adam")
model.fit(x, y, epochs = 150, batch_size = 1, validation_split = 0.2, callbacks = [es, cp, tb])

# 4. 평가, 예측
loss = model.evaluate(x, y)
print("loss: ", loss)

x_pred = x_pred.reshape(1, 3, 1)        # (3,) -> (1, 3, 1)
print(x_pred)

result = model.predict(x_pred)
print("result: ", result)

# loss:  0.4433615803718567
# result:  [[80.322945]]

# loss:  0.5716427564620972
# result:  [[80.266685]]

# Conv1D - Flatten 적용
# loss:  21.85704231262207
# result:  [[[72.561264]]]

# Conv1D - Flatten 미적용
# loss:  57.87166976928711
# result:  [[[66.1933]]]