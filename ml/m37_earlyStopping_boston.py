from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
# x, y = load_boston(return_X_y = True)
datasets = load_boston()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 77)

# 2. 모델
model = XGBRegressor(n_estimators = 1000, learning_rate = 0.01, n_jobs = 8)

# 3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric = ['rmse'],
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 10)
          # 지표가 2개일때 early_stopping의 기준은??
          # 마지막에 적은 지표를 기준으로 적용

RMSE = model.score(x_test, y_test)
print('RMSE: ', RMSE)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('R2: ', r2)
# RMSE:  0.9130109138956524
# R2:  0.9130109138956524

result = model.evals_result()           # 훈련하는 동안의 RMSE 수치
# print('result: ', result)
# result:  {'validation_0': OrderedDict([('rmse', [23.969549, 23.741985, 23.516665, 23.294718, 23.073822, 22.855112, 22.638569, 22.425325, 22.214199, 22.004007])]), 'validation_1': OrderedDict([('rmse', [22.306389, 22.086733, 21.869745, 21.656765, 
# 21.444143, 21.233673, 21.02533, 20.819252, 20.617817, 20.416393])])}