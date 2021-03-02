# 다시 처음부터 작성해보기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras.backend as K

# Function
################################### 'GHI'라는 지표를 추가 ###################################
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

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
q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def quantile_loss(q, y, pred):
  err = (y-pred)
  return K.mean(K.maximum(q*err, (q-1)*err), axis = -1)

# Train Data
file_path = '../data/csv/Sunlight_generation/train/train.csv'
t_dataset = pd.read_csv(file_path, encoding = 'CP949', engine = 'python')

# Graph 확인
# plt.figure(figsize = (20, 15))
# sns.heatmap(data = t_dataset.corr(), annot = True, fmt ='f', linewidths = 2, cmap = 'Blues')
# plt.show()
# 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'

size = 1
t_dataset = preprocessing_data(t_dataset)
x, y = split_xy(t_dataset, size, 8, size, 2)
print(x.shape)