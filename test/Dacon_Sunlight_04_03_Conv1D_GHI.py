import numpy as np
import pandas as pd
from tensorflow.keras.backend import mean, maximum
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 함수 : GHI column 추가
def Add_features(data):
  data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
  data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
  data.drop(['cos'], axis= 1, inplace = True)
  return data

# 함수 정의
def split_x(dataset, is_train = True):
  dataset = Add_features(dataset)
  temp = dataset.copy()
  temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]
  if is_train==True:          
    temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
    temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
    temp = temp.dropna()    # 결측값 제거
    return temp.iloc[:-96]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)
  elif is_train==False:     
    return temp.iloc[-48:, :]   # 0 ~ 6일 중 마지막 6일 데이터만 남긴다. (6일 데이터로 7, 8일을 예측하고자 함)

q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis = -1)

# Train Data 불러오기
file_path1 = '../data/csv/Sunlight_generation/train/train.csv'
train_csv = pd.read_csv(file_path1, engine = 'python', encoding = 'CP949')
t_dataset = split_x(train_csv)
print(t_dataset.columns)
t_dataset = t_dataset.to_numpy()

# Test Data 불러오기
str1 = '../data/csv/Sunlight_generation/test/'
str2 = '.csv'
x_test = []
for i in range(81):
  file_path2 = str1 + str(i) + str2
  test_csv = pd.read_csv(file_path2, engine = 'python', encoding = 'CP949')
  test_csv = split_x(test_csv, False)
  x_test.append(test_csv)
x_test = pd.concat(x_test)
x_test = x_test.to_numpy()

# x, y 분리하기
x_train = t_dataset[:, :8]
y_train1 = t_dataset[:, -2:-1]
y_train2 = t_dataset[:, -1:]
y_train1 = y_train1.reshape(y_train1.shape[0], 1)
y_train2 = y_train2.reshape(y_train2.shape[0], 1)


# Train, Val 분리하기
x_train, x_val, y_train1, y_val1, y_train2, y_val2 = train_test_split(x_train, y_train1, y_train2, train_size = 0.8, shuffle = False)

# MinMaxSclaer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# reshape
x_train = x_train.reshape(x_train.shape[0], 1, 8)
x_val = x_val.reshape(x_val.shape[0], 1, 8)
x_test = x_test.reshape(x_test.shape[0], 1, 8)

# Make Model
def mymodel():
  model = Sequential()
  model.add(Conv1D(filters = 256, kernel_size = 2, strides = 1, padding = 'same', input_shape = (1, 8), activation = 'relu'))
  model.add(Conv1D(filters = 128, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu'))
  model.add(Dense(128, activation = 'relu'))
  model.add(Dense(64, activation = 'relu'))
  model.add(Flatten())
  model.add(Dense(32, activation = 'relu'))
  model.add(Dense(16, activation = 'relu'))
  model.add(Dense(8, activation = 'relu'))
  model.add(Dense(1))
  return model

# Compile, Fit
es = EarlyStopping(monitor = 'loss', patience = 5, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 3)

# y_train1
result1 = []
for q in q_list:
  model = mymodel()
  file_path = "../data/modelcheckpoint/Sunlight/Sunlight_04/Sunlight_04_03_" + str(q) + "_{epoch:02d}.hdf5"
  cp = ModelCheckpoint(filepath = file_path, save_best_only = True, monitor = 'loss')
  model.compile(loss = lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer = 'adam', metrics = 'mae')
  model.fit(x_train, y_train1, epochs = 20, batch_size = 35, validation_data = (x_val, y_val1), callbacks = [es, reduce_lr, cp])
  y_predict1 = pd.DataFrame(model.predict(x_test, batch_size = 35))
  result1.append(y_predict1)
result1 = pd.concat(result1, axis = 1)
result1[result1 < 0] = 0

# y_train2
result2 = []
for q in q_list:
  model = mymodel()
  file_path = "../data/modelcheckpoint/Sunlight/Sunlight_04/Sunlight_04_04_" + str(q) + "_{epoch:02d}.hdf5"
  cp = ModelCheckpoint(filepath = file_path , save_best_only = True, monitor = 'loss')
  model.compile(loss = lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer = 'adam', metrics = 'mae')
  model.fit(x_train, y_train2, epochs = 20, batch_size = 35, validation_data = (x_val, y_val1), callbacks = [es, reduce_lr, cp])
  y_predict2 = pd.DataFrame(model.predict(x_test, batch_size = 35))
  result2.append(y_predict2)
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
