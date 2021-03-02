import numpy as np
from tensorflow.keras.models import load_model
# 데이터
dataset = np.load('../data/npy/Samsung_xy.npz')
x_train = dataset['x_train']
x_test = dataset['x_test']
x_val = dataset['x_val']
y_train = dataset['y_train']
y_test = dataset['y_test']
y_val = dataset['y_val']
x_predict = dataset['x_predict']

# Model Load
model = load_model('../data/modelcheckpoint/Samsung_87_2328623.000000.hdf5')

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

y_test_predict = model.predict(x_test)
rmse = RMSE(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)

loss, mae = model.evaluate(x_test, y_test, batch_size = 16)
print('loss: ', loss)
print('mae: ', mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", r2)
print("y_predict[-1]: ", y_test_predict[-1])
print("y_test[-1]: ", y_test[-1])

result = model.predict(x_predict)
print('result: ', result)