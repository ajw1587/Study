# 다시 처음부터 작성해보자
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

def preprocess_x(dataset, is_train = True):
    temp = dataset.copy()
    temp = Add_features(temp)
    temp = temp.loc[:, ['TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    if is_train == True:
        temp['Target1'] = dataset['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['Target2'] = dataset['TARGET'].shift(-96).fillna(method = 'ffill')
        temp.dropna()
        return temp.iloc[:-96, :]
    else:
        return temp.iloc[-48:, :]

q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis = -1)

def split_xy(dataset):
    temp = dataset.copy()
    x = temp.iloc[:, :-2]
    y = temp.iloc[:, -2:]
    x = x.to_numpy()
    y = y.to_numpy()
    return x, y


# Train Data 가져오기
# print(dataset.columns)                        # ['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET']
csv_file_path = '../data/csv/Sunlight_generation/train/train.csv'
# 상관관계 분석 목표값: Target
# ['DHI', 'DNI', 'WS', 'RH', 'T'] 로 확정
df = pd.read_csv(csv_file_path, engine = 'python', encoding = 'CP949')
# plt.figure(figsize = (15, 10))
# sns.heatmap(data = df.corr(), annot = True, fmt = '.4f', linewidths = .5, cmap = 'Blues')
# plt.show()
df = preprocess_x(df)
x, y = split_xy(df)

# Test Data 가져오기
x_test = []
for i in range(81):
  file_path2 = '../data/csv/Sunlight_generation/test/' + str(i) + '.csv'
  test_csv = pd.read_csv(file_path2, engine = 'python', encoding = 'CP949')
  test_csv = preprocess_x(test_csv, False)
  x_test.append(test_csv)
x_test = pd.concat(x_test, axis = 0)
x_test = x_test.to_numpy()                      # (3888, 4)

# train, val 나누기
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size = 0.8, random_state = 77)

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

print(x_train.shape)        # (41971, 7)
print(y_train.shape)        # (41971, 2)
print(x_test.shape)         # (3888, 7)

x_train = x_train.reshape(x_train.shape[0], 1, 1, x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0], 1, 1, x_val.shape[1])
x_tset = x_train.reshape(x_test.shape[0], 1, 1, x_test.shape[1])


# Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten

input1 = Input(shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3]))
dense1 = Conv2D(filters = 128, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu')(input1)
dense1 = Dense(256, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Flatten()
dense1 = Dense(16, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(8, activation = 'relu')(dense1)
output1 = Dense(2)
model = Model(inputs = input1, outputs = output1)


# Compile and Fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
file_path = '../data/modelcheckpoint/Sunlight/Sunlight_05/Sunlight_05_1_normal'
cp = ModelCheckpoint(file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor = 'val_loss', patience = 8, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 4, mode = 'auto')

model.compile(loss = lambda y_test, y_pred: quantile_loss(q, y_test, y_pred))