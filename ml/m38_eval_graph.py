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
model.fit(x_train, y_train, verbose = 1, eval_metric = ['rmse', 'mae', 'logloss'],
          eval_set = [(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds = 5)

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

# 4. 그래프
import matplotlib.pyplot as plt
epochs = len(result['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, result['validation_1']['logloss'], label = 'Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, result['validation_1']['rmse'], label = 'Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')

fig, ax = plt.subplots()
ax.plot(x_axis, result['validation_0']['mae'], label = 'Train')
ax.plot(x_axis, result['validation_1']['mae'], label = 'Test')
ax.legend()
plt.ylabel('mae')
plt.title('XGBoost MAE')
plt.show()