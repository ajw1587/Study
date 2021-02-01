# m31로 만든 0.95 이상의 n_components = 154를 사용하여
# XGB 모델을 만들것
# mnist dnn 보다 성능 좋게 만들어라!!!

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

# 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape: \n", x_train.shape)               # (60000, 28, 28)
print("y_train.shape: \n", y_train.shape)               # (60000, )
print("x_test.shape: \n", x_test.shape)                 # (10000, 28, 28)
print("y_test.shape: \n", y_test.shape)                 # (10000, )

# PCA 적용
x = np.append(x_train, x_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)
print(x.shape)                                          # (70000, 28, 28)
print(y.shape)                                          # (70000,)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])      # (70000, 784)
pca = PCA(n_components = 154)
x = pca.fit_transform(x)
print(x.shape)

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
print(x_train.shape)
print(x_test.shape)

# 모델 작성
model = Pipeline([('scaler', MinMaxScaler()), ('xgb', XGBClassifier(n_jobs = 8))])

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)

# PCA 사용 전
# loss:  0.14132362604141235
# acc:  0.9690999984741211

# PCA 사용 후
# 0.9632142857142857