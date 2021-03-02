import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, concatenate, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
model = load_model('../data/modelcheckpoint/2-Samsung_Stock_590_726245.0000.hdf5')

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

# loss:  2440800.5
# mae:  955.50146484375
# RMSE:  1562.306108055918
# R2_SCORE:  0.9728562988052274
# y_predict:  [[86461.414 86457.84 ]]
