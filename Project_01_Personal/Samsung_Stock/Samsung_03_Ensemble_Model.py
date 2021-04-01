import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, concatenate, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score

filename1 = '../data/npy/Samsung_StockData.npz'
filename2 = '../data/npy/Samsung_KODEX.npz'

stock = np.load(filename1)
kodex = np.load(filename2)

s_x_train = stock['x_train']          # (690, 5, 8)
s_x_test = stock['x_test']            # (216, 5, 8)
s_x_val = stock['x_val']              # (173, 5, 8)
s_x_predict = stock['x_predict']      # (1, 5, 8)
s_y_train = stock['y_train']          # (690, 2)
s_y_test = stock['y_test']            # (216, 2)
s_y_val = stock['y_val']              # (173, 2)

k_x_train = stock['x_train']          # (690, 5, 8)
k_x_test = stock['x_test']            # (216, 5, 8)
k_x_val = stock['x_val']              # (173, 5, 8)
k_x_predict = stock['x_predict']      # (1, 5, 8)

# =================================================================================
# model1
input1 = Input(shape = (s_x_train.shape[1], s_x_train.shape[2]), name = 'input1')
lstm1 = LSTM(128, activation = "relu")(input1)
dense1 = Dense(256, activation = "relu")(lstm1)
dense1 = Dense(512, activation = "relu")(dense1)
dense1 = Dense(256, activation = "relu")(dense1)
dense1 = Dense(128, activation = "relu")(dense1)
dense1 = Dense(64, activation = "relu")(dense1)

# model2
input2 = Input(shape = (k_x_train.shape[1], k_x_train.shape[2]), name = 'input2')
lstm2 = LSTM(128, activation = "relu")(input2)
dense2 = Dense(256, activation = "relu")(lstm2)
dense2 = Dense(512, activation = "relu")(dense1)
dense2 = Dense(256, activation = "relu")(dense1)
dense2 = Dense(128, activation = "relu")(dense2)
dense2 = Dense(64, activation = "relu")(dense2)

# 모델 병합
from tensorflow.keras.layers import concatenate
merge = concatenate([dense1, dense2])
middle = Dense(256, activation = "relu")(merge)
middle = Dense(128, activation = "relu")(middle)
middle = Dense(64, activation = "relu")(middle)
middle = Dense(32, activation = "relu")(middle)
output1 = Dense(2, name = "output1")(middle)
model = Model(inputs = [input1, input2], outputs = output1)

# Compile, Fit
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

model_path = '../data/modelcheckpoint/Samsung_Stock_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'val_loss', patience =50, mode = 'auto')
cp = ModelCheckpoint(filepath = model_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.5)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
hist = model.fit([s_x_train, k_x_train], s_y_train, batch_size = 24, epochs = 1000, validation_data = ([s_x_val, k_x_val], s_y_val),
                 callbacks = [es, cp, reduce_lr])
loss, mae = model.evaluate([s_x_test, k_x_test], s_y_test, batch_size = 24)

y_test_predict = model.predict([s_x_test, k_x_test])
rmse = RMSE(s_y_test, y_test_predict)
r2 = r2_score(s_y_test, y_test_predict)

print("loss: ", loss)
print("mae: ", mae)
print('RMSE: ', rmse)
print('R2_SCORE: ', r2)

y_predict = model.predict([s_x_predict, k_x_predict])
print('y_predict: ', y_predict)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 6))           # 면적 잡아주기

plt.subplot(2, 1, 1)                    # 2행1열짜리 그래프중 1번째
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()

# plt.title('손실비용')
plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

# loss:  2126523.5
# mae:  1140.1182861328125
# RMSE:  1458.2604710162489
# R2_SCORE:  0.97631215600906
# y_predict:  [[91128.086 91202.21 ]]

# loss:  2335740.25
# mae:  1132.0643310546875
# RMSE:  1528.3128592809408
# R2_SCORE:  0.9739927964886373
# y_predict:  [[93675.06  93697.336]]
