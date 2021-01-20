import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Test Data
test_file_path = '../data/csv/Sunlight_generation/test/0.csv'

test_dataset = pd.read_csv(test_file_path, engine = 'python', encoding = 'CP949')
for i in range(1, 81):
    f_file_path = '../data/csv/Sunlight_generation/test/'
    l_file_path = '.csv'
    file_path = f_file_path + str(i) + l_file_path
    dataset2 = pd.read_csv(file_path, engine = 'python', encoding = 'CP949')
    test_dataset = pd.concat([test_dataset, dataset2])

x_predict = test_dataset.to_numpy()
print(x_predict.shape)                  # (27216, 9)
x_predict = x_predict.reshape(81, 336, 9)
x_predict = x_predict.reshape(81, 48, 9, 7)
print(x_predict.shape)                  # (81, 48, 9, 7)

# model
file_path = "../data/modelcheckpoint/Sunlight_{epoch:02d}_{val_loss:f}.hdf5"
model = load_model(file_path)

y_predict = model.predict(x_predict)
y_predict = y_predict.reshape(7776, 9)
print(y_predict.shape)


df = pd.DataFrame(y_predict)
print(df.shape)
df.to_csv('../data/csv/Sunlight_result.csv')