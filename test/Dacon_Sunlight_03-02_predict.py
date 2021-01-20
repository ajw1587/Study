import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    # 얕은 복사
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    # 필요 컬럼 추출

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train==False:
        return temp.iloc[-48:]

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

# quantile 예측
from tensorflow.keras.backend import mean, maximum

def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

file_path = '../data/modelcheckpoint/Sunlight_03_LSTM_0.1.hdf5'
model = load_model(file_path)
y_predict = model.predict(x_pred_test, batch_size = 7)
print(y_predict[0])