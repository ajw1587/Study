from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
print(dataset)
# print(dataset.keys())             # print(dataset['keys'])
# print(dataset.frame)              # print(dataset['frame'])
# print(dataset.target_names)       # print(dataset['target_names'])
# print(dataset.DESCR)              # print(dataset['DESCR'])
# print(dataset.feature_names)      # print(dataset['feature_names'])

x_data = dataset['data']            # x = dataset.data
y_data = dataset['target']          # y = dataset.target

print(type(x_data), type(y_data))

np.save('../data/npy/iris_x.npy', arr=x_data)        # Study안에 있는 data파일에 x_data 값 저장
np.save('../data/npy/iris_y.npy', arr=y_data)        # Study안에 있는 data파일에 y_data 값 저장