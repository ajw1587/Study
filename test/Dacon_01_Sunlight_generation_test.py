import numpy as np
import pandas as pd

# Test Data
test_file_path = '../data/csv/Sunlight_generation/test/0.csv'
test_dataset = pd.read_csv(test_file_path, engine = 'python', encoding = 'CP949')

for i in range(1, 81):
    f_file_path = '../data/csv/Sunlight_generation/test/'
    l_file_path = '.csv'
    file_path = f_file_path + str(i) + l_file_path
    dataset2 = pd.read_csv(file_path, engine = 'python', encoding = 'CP949')
    test_dataset = pd.concat([test_dataset, dataset2])
print(test_dataset.shape)                # (27216, 9)
#================================================
print(test_dataset.columns)              # Day, Hour, Minute, DHI, DNI, WS, RH, T, TARGET

test_dataset = test_dataset.to_numpy()
test_dataset = test_dataset.reshape(567, 48, 9)
print(test_dataset.shape)              # (567, 48, 9)
test_dataset = test_dataset.reshape(3888, 7, 9)
print(test_dataset.shape)              # (3888, 7, 9)
# 7776

# Evaluate
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

model = load_model("../data/modelcheckpoint/Sunlight_03_3465.892090.hdf5")

y_test_predict = model.predict(test_dataset)
print(y_test_predict.shape)
y_test_predict = y_test_predict.reshape(7776, 9)
print(y_test_predict.shape)
print(type(y_test_predict))

df = pd.DataFrame(y_test_predict)
print(df.shape)
df.to_csv('../data/csv/Sunlight_result.csv')