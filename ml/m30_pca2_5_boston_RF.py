# 차원축소 PCA
# 주성분 분석

import numpy as np
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)                     # (442, 10) (442,)

# pca = PCA(n_components = 9)
# x2 = pca.fit_transform(x)
# print(x2.shape)                             # (442, 10) -> (442, 9)

# pca_EVR = pca.explained_variance_ratio_     # 압축시킨 7개의 Column 중요성 비율
# print(pca_EVR)
# print(sum(pca_EVR))

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum: ', cumsum)       # 밑에 설명
# cumsum:  [0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
#  0.94794364 0.99131196 0.99914395]

d = np.argmax(cumsum >= 0.95) + 1
print('cumsum >= 0.95', cumsum >= 0.95)
print('d: ', d)
# cumsum >= 0.95 [False False False False False False False  True  True  True]
# d:  8

# 그래프
import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()
plt.show()

# 모델
pca2 = PCA(n_components = 8)
x = pca2.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)
model = Pipeline([('scaler', MinMaxScaler()), ('forest', RandomForestRegressor())])

# Fit
model.fit(x_train, y_train)

# Score, Predict
score = model.score(x_test, y_test)
y_pred = model.predict(x_test)
print('score: ', score)

##########################################################################################
# np.cumsum: 각 원소들의 누적 합을 표시함. 각 row와 column의 구분은 없어지고, 순서대로 sum을 함.

# print np.cumsum(a, dtype = float)
# --> 결과 : 
# [  1.   3.   6.  10.  15.  21.]
# --> 설명 : 결과 값의 변수 type을 설정하면서 누적 sum을 함.


# print np.cumsum(a, axis = 0)
# --> 결과 : 
# [[1 2 3]
#  [5 7 9]]
# --> 설명: axis = 0은 같은 column 끼리의 누적 합을 함.

# print np.cumsum(a, axis = 1)
# --> 결과 : 
# [[ 1  3  6]
#  [ 4  9 15]]
# --> 설명 : axis = 1은 같은 row끼리의 누적 합을 함.
##########################################################################################
