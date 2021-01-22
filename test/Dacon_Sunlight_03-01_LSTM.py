import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum


# quantile 지표
def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)


# Target1, Target2 컬럼 추가하기
def preprocess_data(dataset, is_train = True):
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
print(dataset.shape)                        # (52560, 9)

train = preprocess_data(dataset)            # ['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET', 'Target1', 'Target2']

x_train = train.iloc[:, :7]                 # (52464, 7), ['Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T']
y_train1 = train.loc[:, 'Target1']          # (52464, 1), ['Target1']
y_train2 = train.loc[:, 'Target2']          # (52464, 1), ['Target2']
x_train = x_train.to_numpy()
y_train1 = y_train1.to_numpy()
y_train2 = y_train2.to_numpy()
y_train1 = y_train1.reshape(y_train1.shape[0], 1)
y_train2 = y_train2.reshape(y_train2.shape[0], 1)


# Test Data 불러오기
test_file_path = '../data/csv/Sunlight_generation/test/0.csv'
first_file_path = '../data/csv/Sunlight_generation/test/'
last_file_path = '.csv'
x_pred_test = pd.read_csv(test_file_path, engine = 'python', encoding = 'CP949')
x_pred_test = preprocess_data(x_pred_test, is_train = False)
for i in range(1, 81):
    file_path = first_file_path + str(i) + last_file_path
    subset = pd.read_csv(file_path, engine = 'python', encoding = 'CP949')
    subset = preprocess_data(subset, is_train = False)
    x_pred_test = pd.concat([x_pred_test, subset])
x_pred_test = x_pred_test.to_numpy()
# print(type(x_pred_test))      # DataFrame
# print(x_pred_test.shape)      # (3888, 7)


# 전처리
x_train, x_val, y_train1, y_val1, y_train2, y_val2 = train_test_split(x_train, y_train1, y_train2, train_size = 0.8, random_state = 77)
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_pred_test = scaler.transform(x_pred_test)

x_train = x_train.reshape(x_train.shape[0], 1, 7)
x_val = x_val.reshape(x_val.shape[0], 1, 7)
x_pred_test = x_pred_test.reshape(x_pred_test.shape[0], 1, 7)

# print('x_train.shape: ', x_train.shape)                 # (41971, 7)
# print('y_train1.shape: ', y_train1.shape)               # (41971, 1)
# print('y_train2.shape: ', y_train2.shape)               # (41971, 1)
# print('x_pred_test.shape: ', x_pred_test.shape)         # (3888, 7)


# 모델 구성
def mymodel():
    input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
    dense1 = LSTM(128, activation = 'relu')(input1)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(256, activation = 'relu')(dense1)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(128, activation = 'relu')(dense1)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(64, activation = 'relu')(dense1)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(64, activation = 'relu')(dense1)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(64, activation = 'relu')(dense1)
    dense1 = Dropout(0.2)(dense1)
    dense1 = Dense(32, activation = 'relu')(dense1)
    dense1 = Dense(16, activation = 'relu')(dense1)
    dense1 = Dense(8, activation=  'relu')(dense1)
    output1 = Dense(1)(dense1)
    model = Model(inputs = input1, outputs = output1)
    return model


# Compile, Fit
# q_list
q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

es = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5)
# y_train1

result1 = []
for q in q_list:
    model = mymodel()
    file_path = "../data/modelcheckpoint/Sunlight/Sunlight_03/Sunlight_03_1_" + str(q) + "_{epoch:02d}.hdf5"
    cp = ModelCheckpoint(filepath = file_path, monitor = 'loss', save_best_only = True, mode = 'auto')
    model.compile(loss=lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer='adam',
                  metrics = [lambda y_test, y_predict: quantile_loss(q, y_test, y_predict)])
    model.fit(x_train, y_train1, epochs = 150, batch_size = 49, validation_data = (x_val, y_val1), callbacks = [es, reduce_lr, cp])
    y_pred_test = pd.DataFrame(model.predict(x_pred_test, batch_size = 49))
    result1.append(y_pred_test)
result1 = pd.concat(result1, axis = 1)
result1[result1 < 0] = 0

# y_train2
result2 = []
for q in q_list:
    model = mymodel()
    file_path = "../data/modelcheckpoint/Sunlight/Sunlight_03/Sunlight_03_2_" + str(q) + "_{epoch:02d}.hdf5"
    cp = ModelCheckpoint(filepath = file_path, monitor = 'loss', save_best_only = True, mode = 'auto')
    model.compile(loss=lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer='adam',
                  metrics = [lambda y_test, y_predict: quantile_loss(q, y_test, y_predict)])
    model.fit(x_train, y_train2, epochs = 150, batch_size = 49, validation_data = (x_val, y_val2), callbacks = [es, reduce_lr, cp])
    y_pred_test = pd.DataFrame(model.predict(x_pred_test, batch_size = 49))
    result2.append(y_pred_test)
result2 = pd.concat(result2, axis = 1)
result2[result2 < 0] = 0

result = pd.concat([result1, result2])
result.to_csv('../Sunlight/Sunlight_result_01.csv')
result = result.to_numpy()
#==========================================================================================================


# submission.csv 가져오기
df = pd.read_csv('../Sunlight/sample_submission.csv')
df.loc[df.id.str.contains('.csv_'), 'q_0.1':] = result
df.to_csv('../Sunlight/sample_submission_result_03.csv')
