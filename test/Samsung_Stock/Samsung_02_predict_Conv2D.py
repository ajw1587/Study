import numpy as np
from tensorflow.keras.models import load_model

# 데이터 불러오기
dataset = np.load('../data/npy/Samsung_xy_2.npz')

x_train = dataset['x_train']
x_test = dataset['x_test']
x_val = dataset['x_val']
x_predict = dataset['x_predict']
print(x_train.shape)
y_train = dataset['y_train']
y_test = dataset['y_test']
y_val = dataset['y_val']

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], x_predict.shape[2], 1)

# Model 불러오기
file_path = '../data/modelcheckpoint/Samsung_Conv2D_62_1877360.6250.hdf5'
model = load_model(file_path)

# Evaluate
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

# loss:  2089941.25
# mae:  1017.7328491210938
# RMSE:  1445.662605635352
# R2_SCORE:  0.9878880569574859
# y_predict[-1]:  [25412.69]
# y_test[-1]:  26700.0
# result:  [[88018.945]]