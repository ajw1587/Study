# Computer_Vision
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Train 데이터 y -> 0~9, 다중분류
file_path = '../data/csv/Computer_Vision/data/train.csv'
dataset = pd.read_csv(file_path, engine = 'python', encoding = 'CP949', index_col = 0)
# print(type(dataset))        # <class 'numpy.ndarray'>
# print(dataset.shape)        # (2048, 786)

x = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
# print(type(x))              # <class 'numpy.ndarray'>
# print(x.shape)              # (2048, 785)
# print(type(y))              # <class 'numpy.ndarray'>
# print(y.shape)              # (2048,)

# 2-1. 문자 데이터 버리기
x = x[:, 1:]
print(x.shape)                # (2048, 784)
x = x/255.
print(x.shape)                # (2048, 784)

# 3. train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

# 4. 모델
# model = Pipeline([('scaler', MinMaxScaler()), 
#                   ('xgb', XGBClassifier(n_estimator = 1000, n_jobs = 8, learning_rate = 0.0001))])
model = XGBClassifier(n_estimator = 1000, n_jobs = 8, learning_rate = 0.05)

# 5. Fit
model.fit(x_train, y_train, 
          verbose = 1,
          eval_metric = 'mlogloss',
          eval_set = [(x_val, y_val)], 
          early_stopping_rounds = 30)

# 6. Score, Pred
score = model.score(x_test, y_test)
print('score: ', score)
# score:  0.4853658536585366