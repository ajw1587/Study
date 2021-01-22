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


# q_list
q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


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


# Test Data 가져오기
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
# print(type(x_pred_test))      # DataFrame
# print(x_pred_test.shape)      # (3888, 7)


# 모델 구성
input1 = Input(shape = (x_train.shape[1], x_train.shape[2]))
dense1 = Dense(128, activation = 'relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128, activation = 'relu')(input1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation = 'relu')(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(32, activation = 'relu')(dense1)
dense1 = Dense(16, activation = 'relu')(dense1)
dense1 = Dense(8, activation=  'relu')(dense1)
output1 = Dense(2)(dense1)
model = Model(inputs = input1, outputs = output1)


# Compile, Fit
es = EarlyStopping(monitor = 'val_loss', patience = 40, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 20)

result = np.array(range(0, 7776))
result = result.reshape(7776, 1)
for q in q_list:
    model.compile(loss=lambda y_test, y_predict: quantile_loss(q, y_test, y_predict), optimizer='adam')
    model.fit(x_train, y_train, epochs = 200)
    y_pred_test = model.predict(x_pred_test, batch_size = 7)
    # print(y_pred_test.shape)      # (3888, 2)
    first_day = y_pred_test[:,0:1]
    second_day = y_pred_test[:,1:2]
    y_pred_test = np.vstack((first_day, second_day))
    result = np.hstack((result, y_pred_test))

#==========================================================================================================


print(result[0])
print(result.shape)                 # (7666, 10)
result = result[:,1:]
result = pd.DataFrame(result)
print(result.index)
print(result.columns)
print(type(result))
result.to_csv('../Sunlight/Sunlight_result_01.csv')

# submission.csv 가져오기
df = pd.read_csv('../Sunlight/sample_submission.csv')
df.loc[df.id.str.contains('.csv_'), 'q_0.1':] = result.values
df.to_csv('../Sunlight/sample_submission_result_01.csv')