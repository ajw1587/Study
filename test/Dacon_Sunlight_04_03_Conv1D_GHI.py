import numpy as np
import pandas as pd
from tensorflow.keras.backend import mean, maximum
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Reshape
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
def preprocessing_df(dataset, is_train = True):
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

def split_xy(dataset, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(dataset)):
        if i > len(dataset)-x_row:
            break
        tmp_x = dataset[i:i+x_row, :x_col]
        tmp_y = dataset[i:i+x_row, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def quantile_loss(q, y, pred):
  err = (y-pred)
  return mean(maximum(q*err, (q-1)*err), axis = -1)

# Train Data 불러오기
file_path1 = '../data/csv/Sunlight_generation/train/train.csv'
train_csv = pd.read_csv(file_path1, engine = 'python', encoding = 'CP949')
t_dataset = preprocessing_df(train_csv)
t_dataset = t_dataset.to_numpy()

# Test Data 불러오기
str1 = '../data/csv/Sunlight_generation/test/'
str2 = '.csv'
x_test = []
for i in range(81):
  file_path2 = str1 + str(i) + str2
  test_csv = pd.read_csv(file_path2, engine = 'python', encoding = 'CP949')
  test_csv = preprocessing_df(test_csv, False)
  x_test.append(test_csv)
x_test = pd.concat(x_test).values
x_test = x_test.reshape(81, 48, 8)

# x, y 분리하기
x_train, y_train = split_xy(t_dataset, 48,8,48,2)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

# Train, Val 분리하기
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False)

# MinMaxSclaer
x_train = x_train.reshape(x_train.shape[0], 48*8)
x_val = x_val.reshape(x_val.shape[0], 48*8)
x_test = x_test.reshape(x_test.shape[0], 48*8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# reshape
x_train = x_train.reshape(x_train.shape[0], 48, 8)
x_val = x_val.reshape(x_val.shape[0], 48, 8)
x_test = x_test.reshape(x_test.shape[0], 48, 8)

# Make Model

model = Sequential()
model.add(Conv1D(256, 2, input_shape=(x_train.shape[1], x_train.shape[2]), padding='same', activation='relu'))
model.add(Conv1D(128, 2, padding='same'))
model.add(Conv1D(64, 2, padding='same'))
model.add(Conv1D(32, 2, padding='same'))
model.add(Flatten())
model.add(Dense(144))
model.add(Dense(96))
model.add(Dense(96))
model.add(Reshape((48,2)))
model.add(Dense(2))

# model = Sequential()
# model.add(Conv1D(filters = 256, kernel_size = 2, padding = 'same', input_shape = (48, 8), activation = 'relu'))
# model.add(Conv1D(filters = 128, kernel_size = 2, padding = 'same', activation = 'relu'))
# model.add(Conv1D(64, 2, activation = 'relu'))
# model.add(Conv1D(32, 2, activation = 'relu'))
# model.add(Conv1D(16, 2, activation = 'relu'))
# model.add(Flatten())
# model.add(Dense(128, activation = 'relu'))
# model.add(Dense(96, activation = 'relu'))
# model.add(Reshape((48, 2)))
# model.add(Dense(2))
# model.summary()

# Compile, Fit
es = EarlyStopping(monitor = 'loss', patience = 15, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 8)

# y_train
result = []
for q in q_list:
  file_path = "../data/modelcheckpoint/Sunlight/Sunlight_04/Sunlight_04_03_" + str(q) + "_{epoch:02d}_{val_loss:.4f}.hdf5"
  cp = ModelCheckpoint(filepath = file_path, save_best_only = True, monitor = 'loss')
  model.compile(loss = lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer = 'adam', metrics = ['mae'])
  model.fit(x_train, y_train, epochs = 1, batch_size = 40, validation_data = (x_val, y_val), callbacks = [es, reduce_lr])
  pred = model.predict(x_test)
  pred = pd.DataFrame(pred.reshape(7776, 1))
  result.append(pred)
result = pd.concat(result, axis = 1)
result[result < 0] = 0
y_predict = result.to_numpy()

print(y_predict.shape)      # (7776, 9)
#==========================================================================================================
# submission.csv 가져오기
df = pd.read_csv('../Sunlight/sample_submission.csv')

for i in range(1,10):
    column_name = 'q_0.' + str(i)
    df.loc[df.id.str.contains('.csv_Day'), column_name] = y_predict[:,i-1].round(2)
    
df.to_csv('../Sunlight/Sunlight_result_01.csv', index = False)
