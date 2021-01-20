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
dataset = pd.read_csv(test_file_path, engine = 'python', encoding = 'CP949')
dataset = preprocess_data(dataset, is_train = False)
for i in range(1, 81):
    file_path = first_file_path + str(i) + last_file_path
    subset = pd.read_csv(file_path, engine = 'python', encoding = 'CP949')
    subset = preprocess_data(subset, is_train = False)
    dataset = pd.concat([dataset, subset])
# print(type(dataset))      
# print(dataset.shape)      # (3888, 7)

model = load_model(filepath)