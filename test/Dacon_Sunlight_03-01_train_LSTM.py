import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

# Target1, Target2 컬럼 추가하기
def preprocess_data(data, is_train = True):
    temp = dataset.copy()
    temp = temp[['Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if(is_train == True):
        temp['Target1'] = dataset['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['Target2'] = dataset['TARGET'].shift(-96).fillna(method = 'ffill')
        return temp.iloc[:-96]
    else:
        return temp.iloc[-48:]

# train data 불러오기
csv_file_path = '../data/csv/Sunlight_generation/train/train.csv'
dataset = pd.read_csv(csv_file_path, engine = 'python', encoding = 'CP949')
print(dataset.shape)                # (52560, 9)

train = preprocess_data(dataset)    # ['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET', 'Target1', 'Target2']

x_train = train.iloc[:, :7]         # (52464, 7), ['Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T']
y_train = train.iloc[:, -2:]        # (52464, 2), ['Target1', 'Target2']
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

x_train = x_train.reshape(1093, 48, 7)
y_train = y_train.reshape(1093, 48, 2)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

# print(x_train.shape)                # (699, 48, 7)
# print(y_train.shape)                # (699, 48, 2)
# print(x_test.shape)                 # (219, 48, 7)
# print(y_test.shape)                 # (219, 48, 2)
# print(x_val.shape)                  # (175, 48, 7)
# print(y_val.shape)                  # (175, 48, 2)

# 모델 구성
input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
dense1 = Dense(128, activation = 'relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
dense1 = Dense(8, activation=  'relu')(dense1)
output1 = Dense(2)(dense1)
model = Model(inputs = input1, outputs = output1)
model.summary()

# Compile, Fit
# file_path = "../data/modelcheckpoint/Sunlight_03_{epoch:02d}_{val_loss:f}.hdf5" 
# cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 15)

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
model.fit(x_train, y_train, batch_size = 14, epochs = 1000, validation_data = (x_val, y_val), callbacks = [es, reduce_lr])

# Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size = 14)
y_predict = model.predict(x_test)

print('loss: ', loss)
print('mae: ', mae)
print(y_predict.shape)
print(y_test.shape)
print(y_predict[0])
print(y_test[0])

#==========================================================================================================
# y_test와 y_predict로 quantile_loss 모델 만들기
from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)


q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for q in q_lst:
    file_path = "../data/modelcheckpoint/Sunlight_03_LSTM_{epoch:02d}.hdf5" 
    cp = ModelCheckpoint(filepath = file_path, monitor = 'loss', save_best_only = True, mode = 'auto')
    model2 = Sequential()
    model2.add(Dense(10))
    model2.add(Dense(1))
    model2.compile(loss=lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer='adam')
    model2.fit(x_train, y_train, epochs = 100, callbacks = ['cp'])