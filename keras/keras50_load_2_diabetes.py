import numpy as np

x = np.load('../data/npy/diabetes_x.npy')
y = np.load('../data/npy/diabetes_y.npy')

print(x)
print(y)
print(x.shape, y.shape)

# 모델을 완성하시오.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, test_size = 0.2, random_state = 66)

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape = (10,))
dense1 = Dense(100, activation = "relu")(input1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(300, activation = "relu")(dense1)
dense1 = Dense(200, activation = "relu")(dense1)
dense1 = Dense(100, activation = "relu")(dense1)
output1 = Dense(1,  activation = "relu")(dense1)
model = Model(inputs = input1, outputs = output1)

# Compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint', monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = "mse", optimizer = "adam", metrics = "mae")
model.fit(x_train, y_train, batch_size = 5, epochs = 300, validation_data = (x_val, y_val), callbacks = [es, cp, tb])

# 평가 및 예측
loss, mae = model.evaluate(x_test, y_test, batch_size = 6)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("loss: ", loss)
print("mae: ", mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", r2)