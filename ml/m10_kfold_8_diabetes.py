import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# 분류에서는 score가 accuracy_score로 잡힌다.
# 1. 데이터
# x, y = load_iris(return_X_y= True)        x, y를 나눠주는 또다른 방법

dataset = load_diabetes()
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
# KNeighborsRegressor() scores:  [ 0.4261522   0.61023501 -0.03631996  0.36239817  0.40768521]
# LinearRegression() scores:  [0.37453315 0.58276145 0.41821927 0.60654312 0.42687594]
# DecisionTreeRegressor() scores:  [-0.11907348  0.0424768   0.01989321 -0.232401   -0.21001174]
# RandomForestRegressor() scores:  [ 0.3017371   0.44625913  0.49484849  0.56210849 -0.02259909]