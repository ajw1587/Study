import numpy as np
import pandas as pd

# # Train Data
csv_file_path = '../data/csv/Sunlight_generation/train/train.csv'
dataset = pd.read_csv(csv_file_path, engine = 'python', encoding = 'CP949')

def make_xy(dataset, idx):
    x = []
    y = []
    for i in range(dataset.shape[0] - idx - 1):
        x_subset = dataset[i:i+idx, :]
        y_subset = dataset[i+idx : i+idx+2]
        x.append(x_subset)
        y.append(y_subset)
    return np.array(x), np.array(y)

print(dataset.shape)              # (52560, 9)
print(dataset.columns)            # Day, Hour, Minute, DHI, DNI, WS, RH, T, TARGET

dataset = dataset.to_numpy()
x, y = make_xy(dataset, 7)
print(x.shape)                    # (52552, 7, 9)
print(y.shape)                    # (52552, 2, 9)

# 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
# x = x.reshape(y.shape[0], y.shape[1]*y.shape[2])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, shuffle = False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False)
print(x_train.shape)            # (33632, 7, 9)
print(x_test.shape)             # (10511, 7, 9)
print(x_val.shape)              # (8409, 7, 9)

print(y_train.shape)            # (33632, 2, 9)
print(y_test.shape)             # (10511, 2, 9)
print(y_val.shape)              # (8409, 2, 9)

y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2])
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1]*y_val.shape[2])

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# x_val = scaler.transform(x_val)


# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten

input1 = Input(shape = (x.shape[1], x.shape[2]))
dense1 = LSTM(512, activation = 'relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(256, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
dense1 = Dense(8, activation = 'relu')(dense1)
output1 = Dense(18)(dense1)
model = Model(inputs = input1, outputs = output1)
model.summary()

# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
file_path = "../data/modelcheckpoint/Sunlight_{epoch:02d}_{val_loss:f}.hdf5"
es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 5, verbose = 1)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size = 36, validation_data = (x_val, y_val), callbacks = [es, cp, reduce_lr])

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

y_test_predict = model.predict(x_test)
rmse = RMSE(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)

loss, mae = model.evaluate(x_test, y_test, batch_size = 36)
print('loss: ', loss)
print('mae: ', mae)
print("RMSE: ", rmse)
print("R2_SCORE: ", r2)
