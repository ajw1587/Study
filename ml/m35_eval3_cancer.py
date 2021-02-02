# 이중분류 모델 완성할 것
# eval_metric 부분 수정

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
# x, y = load_breast_cancer(return_X_y = True)
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 77)

# 2. 모델
model = XGBClassifier(n_estimators = 10, learning_rate = 0.01, n_jobs = 8)

# 3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric = 'logloss',
          eval_set = [(x_train, y_train), (x_test, y_test)])

logloss = model.score(x_test, y_test)
print('logloss: ', logloss)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy: ', accuracy)
# logloss:  0.9298245614035088
# accuracy:  0.9298245614035088

result = model.evals_result()           
print('result: ', result)
# result:  {'validation_0': OrderedDict([('logloss', [0.684247, 0.675587, 0.667092, 0.658756, 
# 0.650576, 0.642548, 0.634666, 0.626927, 0.619327, 0.611861])]), 
# 'validation_1': OrderedDict([('logloss', [0.685169, 0.67734, 0.669669, 0.662153, 0.654787, 
# 0.647568, 0.640439, 0.633421, 0.626563, 0.619866])])}