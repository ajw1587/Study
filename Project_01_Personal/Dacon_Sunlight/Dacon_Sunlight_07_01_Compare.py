# from google.colab import drive
# drive.mount('/content/gdrive/')
# Google Drive 경로: /content/gdrive/My Drive/Colab Notebooks/

# LSTM, GRU, Conv1D 비교하기

# 다시 처음부터 작성해보기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K

# 1. Function
################################### 'GHI'라는 지표를 추가 ###################################
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

################################### '전일사량' 지표를 추가 ###################################
# 전일사량
# 일사량은, Rc=(0.29 + 0.71(1-Ro) 이다.
# 여기서 Ro는 알베도를 말하는 건데... 즉, 이말은 "전일사량 = 직달일사량 + 산란일사량

################################### Data Column 및 Y값 Column생성 ###################################
def preprocessing_data(dataset, is_train = True):
    temp = dataset.copy()
    temp = Add_features(temp)
    temp = temp[['Hour', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET']]
    if is_train == True:
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['Target2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp.dropna()
        return np.array(temp.iloc[:-96, :])
    else:
        return np.array(temp.iloc[-48:, :])

################################### x, y split ###################################
def split_xy(dataset, x_row, x_column, y_row, y_column):
    x, y = list(), list()
    for i in range(len(dataset)-x_row+1):
        subset_x = dataset[i: i + x_row, :x_column]
        subset_y = dataset[i: i + y_row, -1*y_column:]
        x.append(subset_x)
        y.append(subset_y)
    return np.array(x), np.array(y)

################################### loss 지표 생성 ###################################
q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  #
def quantile_loss(q, y, pred):
  err = (y-pred)
  return K.mean(K.maximum(q*err, (q-1)*err), axis = -1)


# 2. Train Data
################################### 파일 불러오기 ###################################
file_path = '/content/gdrive/My Drive/Sunlight_generation/train/train.csv'
t_dataset = pd.read_csv(file_path, encoding = 'CP949', engine = 'python')

# Graph 확인
# plt.figure(figsize = (20, 15))
# sns.heatmap(data = t_dataset.corr(), annot = True, fmt ='f', linewidths = 2, cmap = 'Blues')
# plt.show()
# 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'

################################### x,y 자르기 ###################################
size = 1
t_dataset = preprocessing_data(t_dataset)
x, y = split_xy(t_dataset, size, 8, size, 2)
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])
# print(x.shape)      # (52464, 1, 8)
# print(y.shape)      # (52464, 2)

################################### y1(day1), y2(day2) 나누기 ###################################
y1 = y[:, 0]
y2 = y[:, 1]
y1 = y1.reshape(y1.shape[0], 1)         # (52464, 1)
y2 = y2.reshape(y2.shape[0], 1)         # (52464, 1)

################################### train, val 나누기 ###################################
from sklearn.model_selection import train_test_split
x_train, x_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x, y1, y2, train_size = 0.8, shuffle = False)
x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x_train, y1_train, y2_train, train_size = 0.8, shuffle = False)

# print(x_train.shape)        # (41971, 1, 8)
# print(x_val.shape)          # (10493, 1, 8)
# print(y1_train.shape)       # (41971, 1)
# print(y1_val.shape)         # (10493, 1)
# print(y2_train.shape)       # (41971, 1)
# print(y2_val.shape)         # (10493, 1)

################################### test 불러오기 (0.csv ~ 80.csv) ###################################
x_pred = list()
for i in range(81):
  file_path = '/content/gdrive/My Drive/Sunlight_generation/test/' + str(i) + '.csv'
  subset = pd.read_csv(file_path, engine = 'python', encoding = 'CP949')
  subset = preprocessing_data(subset, False)
  x_pred.append(subset)
x_pred = np.array(x_pred)     # (81, 48, 8)
x_pred = x_pred.reshape(x_pred.shape[0]*x_pred.shape[1], size, size, x_pred.shape[2])

################################### MinMaxScaler 전처리 ###################################
# from sklearn.preprocessing import MinMaxScaler
# x_train = x_train.reshape(x_train.shape[0]* x_train.shape[1], x_train.shape[2])
# x_val = x_val.reshape(x_val.shape[0]* x_val.shape[1], x_val.shape[2])
# x_test = x_test.reshape(x_test.shape[0]* x_test.shape[1], x_test.shape[2])

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_val = scaler.transform(x_val)
# x_test = scaler.transform(x_test)

# x_train = x_train.reshape(x_train.shape[0], size, x_train.shape[1])
# x_val = x_val.reshape(x_val.shape[0], size, x_val.shape[1])
# x_test = x_test.reshape(x_test.shape[0], size, x_test.shape[1])

# # print(x_train.shape)          # (41971, 1, 8)
# # print(x_val.shape)            # (10493, 1, 8)
# # print(x_test.shape)           # (3888, 1, 8)
# # print(y1_train.shape)         # (41971, 1)
# # print(y1_val.shape)           # (10493, 1)
# # print(y2_train.shape)         # (41971, 1)
# # print(y2_val.shape)           # (10493, 1)

################################### MinMaxScaler 전처리 ###################################
from sklearn.preprocessing import StandardScaler
x_train = x_train.reshape(x_train.shape[0]* x_train.shape[1], x_train.shape[2])
x_val = x_val.reshape(x_val.shape[0]* x_val.shape[1], x_val.shape[2])
x_test = x_test.reshape(x_test.shape[0]* x_test.shape[1], x_test.shape[2])

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], size, size, x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0], size, size, x_val.shape[1])
x_test = x_test.reshape(x_test.shape[0], size, size, x_test.shape[1])

# print(x_train.shape)          # (41971, 1, 8)
# print(x_val.shape)            # (10493, 1, 8)
# print(x_test.shape)           # (3888, 1, 8)
# print(y1_train.shape)         # (41971, 1)
# print(y1_val.shape)           # (10493, 1)
# print(y2_train.shape)         # (41971, 1)
# print(y2_val.shape)           # (10493, 1)

################################### Model ###################################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GRU, Conv1D, Conv2D, LSTM, Dense, Input, Dropout, Flatten

def my_model():
  input1 = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
  dense1 = Conv2D(256, 2, padding = 'same', activation = 'relu')(input1)
  dense1 = Dropout(0.2)(dense1)
  dense1 = Dense(128, activation = 'relu')(dense1)
  dense1 = Dropout(0.2)(dense1)
  dense1 = Dense(128, activation = 'relu')(dense1)
  dense1 = Dropout(0.2)(dense1)
  dense1 = Flatten()(dense1)
  dense1 = Dense(64, activation = 'relu')(dense1)
  dense1 = Dropout(0.2)(dense1)
  dense1 = Dense(32, activation = 'relu')(dense1)
  dense1 = Dropout(0.2)(dense1)
  dense1 = Dense(16, activation = 'relu')(dense1)
  dense1 = Dropout(0.2)(dense1)
  output1 = Dense(1)(dense1)
  model = Model(inputs = input1, outputs = output1)
  model.summary()
  return model

################################### Compile and Fit and Predict ###################################
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor = 'val_loss', patience = 50, mode = 'min')
cp = ModelCheckpoint(file_path, monitor = 'val_loss', save_best_only = True, verbose = 1, mode = 'min')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 25, factor = 0.5, mode = 'min')
model = my_model()

def compile_model(model, x_train, y_train, x_val, y_val, x_test, y_test, x_pred):
  loss_list = list()
  pred_list = list()
  for q in q_list:
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer = 'adam')
    model.fit(x_train, y_train, batch_size = 56, epochs = 500, validation_data = (x_val, y_val), callbacks = [es, reduce_lr])
    
    # Evaluate and Predict
    loss = model.evaluate(x_test, y_test, batch_size = 56)
    pred = model.predict(x_pred)

    loss_list.append(loss)
    pred_list.append(pred)

  fin_loss = sum(loss_list)/len(loss_list)
  fin_pred = np.array(pred_list)
  fin_pred = fin_pred.reshape(3888, 9)
  return fin_pred, fin_loss

################################### Predict ###################################
y1_pred, y1_loss = compile_model(model, x_train, y1_train, x_val, y1_val, x_test, y1_test, x_pred)
y2_pred, y2_loss = compile_model(model, x_train, y2_train, x_val, y2_val, x_test, y2_test, x_pred)

print(y1_loss)
print(y2_loss)
y1_pred = pd.DataFrame(y1_pred)
y1_pred[y1_pred < 0] = 0
y2_pred = pd.DataFrame(y2_pred)
y2_pred[y2_pred < 0] = 0

################################### to_csv ###################################
result = pd.read_csv('/content/gdrive/My Drive/Sunlight_generation/sample_submission.csv', engine = 'python', encoding = 'CP949')
result.loc[result.id.str.contains('Day7'), 'q_0.1':] = y1_pred
result.loc[result.id.str.contains('Day8'), 'q_0.1':] = y2_pred
result.to_csv('/content/gdrive/My Drive/test/Sunlight_0125_Conv1D01.csv')

# GRU
# 2.328190247217814
# 2.408152461051941

# LSTM
# 2.3439280721876354
# 2.4437080224355063

# Conv1D
# 2.310122741593255
# 2.3622536526785956