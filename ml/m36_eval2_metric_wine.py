# 다중분류 metric을 3개 이상 넣어서 학습

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

# 1. 데이터
# x, y = load_wine(return_X_y = True)
datasets = load_wine()
x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 77)

# 2. 모델
model = XGBClassifier(n_estimators = 100, learning_rate = 0.01, n_jobs = 8)

# 3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric = ['mlogloss', 'merror', 'cox-nloglik'],
          eval_set = [(x_train, y_train), (x_test, y_test)])

mlogloss = model.score(x_test, y_test)
print('mlogloss: ', mlogloss)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('accuracy: ', accuracy)
# ERROR:  0.9166666666666666
# accuracy:  0.9166666666666666

result = model.evals_result()           
# print('result: ', result)
# result:  {'validation_0': OrderedDict([('mlogloss', [1.085057, 
# 1.071729, 1.058624, 1.045737, 1.033061, 1.020591, 1.008428, 0.996461, 0.984684, 0.973093])]), 
# 'validation_1': OrderedDict([('mlogloss', [1.085895, 1.073394, 1.061104, 1.049021, 1.037188, 
# 1.025501, 1.01409, 1.002863, 0.991817, 0.980947])])}