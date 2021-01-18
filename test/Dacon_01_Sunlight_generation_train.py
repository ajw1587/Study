import numpy as np
import pandas as pd

# # Train Data
# csv_file_path = '../data/csv/Sunlight_generation/train/train.csv'
# dataset = pd.read_csv(csv_file_path, engine = 'python', encoding = 'CP949')

def make_xy(dataset, idx):
    x = []
    y = []
    for i in range(dataset.shape[0] - idx - 1):
        x_subset = dataset[i:i+idx, :]
        y_subset = dataset[i+idx : i+idx+2, :]
        x.append(x_subset)
        y.append(y_subset)
    return np.array(x), np.array(y)

# print(dataset.shape)              # (52560, 9)
# print(dataset.columns)            # Day, Hour, Minute, DHI, DNI, WS, RH, T, TARGET

# dataset = dataset.to_numpy()
# dataset = dataset.reshape(1095, 48, 9)
# print(dataset.shape)              # (1095, 48, 9)
# dataset = dataset.reshape(dataset.shape[0], dataset.shape[1]*dataset.shape[2])
# print(dataset.shape)              # (1095, 432) -> x 7개씩, y 2개씩

# x, y = make_xy(dataset, 7)

# print(x.shape)                    # (1087, 7, 432)
# print(y.shape)                    # (1087, 2, 432)

# Test Data
test_file_path = '../data/csv/Sunlight_generation/test/0.csv'
dataset = pd.read_csv(test_file_path, engine = 'python', encoding = 'CP949')

for i in range(1, 81):
    f_file_path = '../data/csv/Sunlight_generation/test/'
    l_file_path = '.csv'
    file_path = f_file_path + str(i) + l_file_path
    dataset2 = pd.read_csv(file_path, engine = 'python', encoding = 'CP949')
    dataset = pd.concat([dataset, dataset2])
print(dataset.shape)                # (27216, 9)
#================================================
print(dataset.columns)              # Day, Hour, Minute, DHI, DNI, WS, RH, T, TARGET

dataset = dataset.to_numpy()
# dataset = dataset.reshape(1095, 48, 9)
# print(dataset.shape)              # 
# dataset = dataset.reshape(dataset.shape[0], dataset.shape[1]*dataset.shape[2])
# print(dataset.shape)              # 

# x, y = make_xy(dataset, 7)

# print(x.shape)                    # 


# 모델 구성