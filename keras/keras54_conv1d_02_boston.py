# Conv1D 활용

import numpy as np
from sklearn.datasets import load_boston

# 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target
print(x.shape)      # (506, 13)
print(y.shape)      # (506,)
print("=========================")
print(x[:5])
print(y[:10])

print(np.max(x), np.min(x))
print(dataset.feature_names)

# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 66)
print(x_train.shape)
print(y_train.shape)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten

model = Sequential()
model.add(Conv1D(10, 2, 1, padding = 'same',input_shape = (x_train.shape[1],1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(100, 2, 1, padding = 'same'))
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(300, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(500, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(300, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10,  activation = "relu"))
model.add(Flatten())
model.add(Dense(1))

# compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
file_path = '../data/modelCheckpoint/k46_4_boston_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, batch_size = 3, epochs = 200, validation_data = (x_val, y_val),
          callbacks = [es])

# 평가 및 예측
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
y_predict = model.predict(x_test)
rmse = RMSE(y_test, y_predict)
R2_SCORE = r2_score(y_test, y_predict)
loss, mae = model.evaluate(x_test, y_test, batch_size = 3)

print("loss(mse): ", loss)
print("mae: ", mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2_SCORE)

# loss(mse):  6.467247009277344
# mae:  1.866719126701355
# RMSE:  2.5430781449016275
# R2_SCORE:  0.9226248004972167

# Conv1D 적용
# loss(mse):  16.378923416137695
# mae:  2.856229305267334
# RMSE:  4.047088366619257
# R2_SCORE:  0.8040398582523338