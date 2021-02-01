# 1.0 이상

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.append(x_train, x_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 1.0) + 1
# print('d: ', d)
# d: 713

pca = PCA(n_components = 713)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)
print(x_train.shape)
print(y_train.shape)

# 모델
parameters = [
    {'C': [1, 1000], 'kernel': ['linear']},                                # 4번
    {'C': [1, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},          # 6번
    {'C': [1, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]} # 8번
]
kfold = KFold(n_splits = 5, shuffle = True)
model = GridSearchCV(XGBClassifier(n_jobs = 8, user_label_encoder = False, n_estimators = 1),
                     parameters, cv = kfold)

# Fit
model.fit(x_train, y_train, eval_metric = 'mlogloss',
          eval_set = [(x_val, y_val)])

# Score
score = model.score(x_test, y_test)
print('score: ', score)