import numpy as np
import pandas as pd

# Train Data
csv_file_path = '../data/csv/Sunlight_generation/train/train.csv'
dataset = pd.read_csv(csv_file_path, engine = 'python', encoding = 'CP949')

def split_xy2(dataset, x_low, x_col, y_low, y_col):
    x, y = [], []
    for i in range(0, len(dataset) - x_low - y_low, 48):
        x_subset = dataset[i : i + x_low, 0 : x_col]
        x.append(x_subset)
    # for i in range(len(dataset) - y_low -1):
    #     if (i + x_low + y_low) >= len(dataset):
    #         break
        y_subset = dataset[i + x_low : i + x_low + y_low , -y_col:]
        y.append(y_subset)
    x = np.array(x)
    y = np.array(y)
    return x, y

# print(dataset.iloc[48:96])

# print(dataset.shape)              # (52560, 9)
# print(dataset.columns)            # Day, Hour, Minute, DHI, DNI, WS, RH, T, TARGET

dataset = dataset.to_numpy()
x, y = split_xy2(dataset, 336, 9, 96, 9)
print(x.shape)                      # (1086, 336, 9)
print(y.shape)                      # (1086, 96, 9) 
x = x.reshape(1086, 48, 9, 7)       # (1086, 336, 9) -> (1086, 7, 48, 9) -> (1086, 48, 9, 7)
y = y.reshape(1086, 48, 9, 2)       # (1086, 96, 9)  -> (1086, 2, 48, 9) -> (1086, 48, 9, 2)

# 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8)

# 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv2D, MaxPooling2D

input1 = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
dense1 = Conv2D(filters = 100, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Conv2D(filters = 100, kernel_size = (2, 2), strides = (1, 1), padding = 'same')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(256, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(256, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
output1 = Dense(2)(dense1)
model = Model(inputs = input1, outputs = output1)
model.summary()


# Compile and Fit and EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
file_path = "../data/modelcheckpoint/Sunlight_02_{epoch:02d}_{val_loss:f}.hdf5"
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 25, verbose = 1)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, epochs = 5000, batch_size = 9, validation_data = (x_val, y_val), callbacks = [es, cp, reduce_lr]) # 

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

loss, mae = model.evaluate(x_test, y_test, batch_size = 9)
print('loss: ', loss)
print('mae: ', mae)

# print("RMSE: ", rmse)
# print("R2_SCORE: ", r2)
