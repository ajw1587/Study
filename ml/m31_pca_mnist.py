import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

(x_train, _), (x_test, _) = mnist.load_data()

x = np.append(x_train, x_test, axis = 0)
print(x.shape)          # (70000, 28, 28)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
print(x.shape)          # (70000, 784)
# 실습
# PCA를 통해 0.95 이상인거 몇개??
# PCA 배운거 다 집어넣고 확인!!

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
# print(sum(pca.explained_variance_ratio_))
# 1.000000000000002
# print('cumsum: ', cumsum)

#########################################cumsum > 0.95
d = np.argmax(cumsum >= 0.95) +1
print('cumsum >= 0.95: ', cumsum >= 0.95)
print('d: ', d)
# cumsum >= 0.95 d:  154

#########################################cumsum > 1.0
# d = np.argmax(cumsum >= 1.0) +1
# print('cumsum >= 1.0: ', cumsum >= 1.0)
# print('d: ', d)
# cumsum >= 1.0  d:  713
