import numpy as np
import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression # 이름만 회귀이고 분류모델이다
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# 분류에서는 score가 accuracy_score로 잡힌다.
# 1. 데이터
# x, y = load_iris(return_X_y= True)        x, y를 나눠주는 또다른 방법

dataset = load_boston()
x = dataset.data
y = dataset.target
# print(y)            # 0, 1, 2

# 2. k-fold 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 77, shuffle = True, train_size = 0.8)
kfold = KFold(n_splits = 5, shuffle = True)     # n_splits: 몇도막으로 나누겠다.

# 3. 모델
model = [KNeighborsRegressor(),
         LinearRegression(),          # 2진분류라 warning 발생
         DecisionTreeRegressor(),
         RandomForestRegressor()]          
for i in range(4):
    scores = cross_val_score(model[i], x_train, y_train, cv=kfold)      # model과 Data를 엮어주는 역할
    print(model[i], 'scores: ', scores)                                        # model이 5번 fit해서 값이 5개이다.

'''
# 4. Compile and Train
model.fit(x_train, y_train)

result1 = model.score(x_test, y_test)
print(result1)

y_predict = model.predict(x_test)
result2 = accuracy_score(y_test, y_predict)
print(result2)
'''
# KNeighborsRegressor() scores:  [0.51676871 0.54685201 0.31036247 0.36407098 0.33699741]
# LinearRegression() scores:  [0.75423337 0.6783362  0.51596879 0.69702277 0.76896706]
# DecisionTreeRegressor() scores:  [0.80054739 0.48884763 0.8890778  0.67259077 0.82487409]
# RandomForestRegressor() scores:  [0.88788955 0.85888559 0.91779244 0.5944129  0.90354789]