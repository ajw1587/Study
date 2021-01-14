import numpy as np
from tensorflow.keras.models import load_model
# 데이터
dataset = np.load('../data/npy/Samsung_xy.npz')
x = dataset['x']
y = dataset['y']
x_predict = dataset['x_predict']

# 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x = x.reshape(x.shape[0], 40)
x_predict = x_predict.reshape(1, 40)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_predict = scaler.transform(x_predict)


x_train = x_train.reshape(x_train.shape[0], 5, 8)
x_test = x_test.reshape(x_test.shape[0], 5, 8)
x_val = x_val.reshape(x_val.shape[0], 5, 8)
x_predict = x_predict.reshape(1, 5, 8)

# Model Load
model = load_model('../data/modelcheckpoint/Samsung_42_2308608.750000.hdf5')

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

y_test_predict = model.predict(x_test)
rmse = RMSE(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)

loss, mae = model.evaluate(x_test, y_test, batch_size = 1)
print('loss: ', loss)
print('mae: ', mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", r2)
print("y_predict[0]: ", y_test_predict[0])
print("y_test[0]: ", y_test[0])

result = model.predict(x_predict)
print('result: ', result)
