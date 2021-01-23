import numpy as np
import pandas as pd
from tensorflow.keras.backend import mean, maximum
from tensorflow.keras.models import Sequential, load_model
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
  temp = temp[['TARGET','GHI','DHI','DNI','WS','RH','T']]
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
x_test = []
for i in range(81):
  file_path2 = '../data/csv/Sunlight_generation/test/' + str(i) + '.csv'
  test_csv = pd.read_csv(file_path2, engine = 'python', encoding = 'CP949')
  test_csv = preprocessing_df(test_csv, False)
  x_test.append(test_csv)
x_test = pd.concat(x_test).values
x_test = x_test.reshape(3888, 7)

# x, y 분리하기
size = 1
x_train, y_train = split_xy(t_dataset, size,7, size,2)

print(x_train.shape)        # (52464, 1, 7)  
print(y_train.shape)        # (52464, 1, 2)
print(x_test.shape)         # (3888, 7)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2])

# Train, Val 분리하기
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, shuffle = False)

# MinMaxSclaer
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

print(x_train.shape)        # (41971, 7) 
print(x_val.shape)          # (10493, 7)
print(x_test.shape)         # (3888, 7)
print(y_train.shape)        # (41971, 2)
print(y_val.shape)          # (10493, 2)

x_train = x_train.reshape(x_train.shape[0], size, x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0], size, x_val.shape[1])
x_test = x_test.reshape(x_test.shape[0], size, x_test.shape[1])

# y_train
result = []
for q in q_list:
  file_path = "../data/modelcheckpoint/Sunlight/Sunlight_04/Sunlight_04_03_" + str(q) + ".hdf5"
  model = load_model(file_path, compile = False)
  model.compile(loss = lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer = 'adam', metrics = ['mae'])
  pred = model.predict(x_test)
  pred = np.concatenate((pred[:,0], pred[:,1]), axis = 0)
  pred = pd.DataFrame(pred.reshape(7776, 1))
  result.append(pred)
result = pd.concat(result, axis = 1)
result[result < 0] = 0
result = result.to_numpy()

#==========================================================================================================
# submission.csv 가져오기
df = pd.read_csv('../Sunlight/sample_submission.csv')

for i in range(1,10):
    column_name = 'q_0.' + str(i)
    df.loc[df.id.str.contains('.csv_Day'), column_name] = result[:,i-1].round(2)
    
df.to_csv('../Sunlight/Sunlight_result_predict.csv', index = False)

# # y_train1
# result1 = []
# for q in q_list:
#   file_path = "../data/modelcheckpoint/Sunlight/Sunlight_04/Sunlight_04_2_GHI/Sunlight_04_3_" + str(q) + ".hdf5"
#   model = load_model(file_path, compile = False)
#   cp = ModelCheckpoint(filepath = file_path, save_best_only = True, monitor = 'loss')
#   model.compile(loss = lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer = 'adam',
#                 metrics = [lambda y_test, y_predict: quantile_loss(q, y_test, y_predict)])
#   y_predict1 = pd.DataFrame(model.predict(x_test, batch_size = 35))
#   result1.append(y_predict1)
# result1 = pd.concat(result1, axis = 1)
# result1[result1 < 0] = 0

# # y_train2
# result2 = []
# for q in q_list:
#   file_path = "../data/modelcheckpoint/Sunlight/Sunlight_04/Sunlight_04_2_GHI/Sunlight_04_4_" + str(q) + ".hdf5"
#   model = load_model(file_path, compile = False)
#   model.compile(loss = lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer = 'adam',
#                 metrics = [lambda y_test, y_predict: quantile_loss(q, y_test, y_predict)])
#   y_predict2 = pd.DataFrame(model.predict(x_test, batch_size = 35))
#   result2.append(y_predict2)
# result2 = pd.concat(result2, axis = 1)
# result2[result2 < 0] = 0

# result = pd.concat([result1, result2])
# result.to_csv('../Sunlight/Sunlight_result_04_04.csv')
# result = result.to_numpy()
# #==========================================================================================================


# # submission.csv 가져오기
# df = pd.read_csv('../Sunlight/sample_submission.csv')
# df.loc[df.id.str.contains('.csv_'), 'q_0.1':] = result
# df.to_csv('../Sunlight/sample_submission_result_03.csv')
