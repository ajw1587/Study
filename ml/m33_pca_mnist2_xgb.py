# m31로 만든 1.0 이상의 n_components = 713를 사용하여
# XGB 모델을 만들것
# mnist dnn 보다 성능 좋게 만들어라!!!

import numpy as np
from tensorflow.keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)
# print(x.shape)      # (70000, 28, 28)
# print(y.shape)      # (70000, )
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

# 2. PCA로 적절한 열 찾기
# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(type(cumsum))
# print(cumsum.shape)
# d = np.argmax(cumsum >= 1.0) + 1
# print('d: ', d)
# 713

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)

# 3. 모델
model = Pipeline([('scaler', MinMaxScaler()), ('xgb', XGBClassifier(n_jobs = 8, user_label_encoder = False))])

# 4. Fit
model.fit(x_train, y_train)

# 5. Score
score = model.score(x_test, y_test)
print('score: ', score)
# score:  0.978