# 0.95 이상
import numpy as np
from tensorflow.keras.datasets import mnist
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.decomposition import PCA


# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x = np.append(x_train, x_test, axis = 0)
y = np.append(y_train, y_test, axis = 0)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

# pca = PCA()
# pca.fit_transform(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum > 0.95) + 1
# print('cumsum >= 0.95: \n', cumsum >= 0.95)
# print('d: ', d)
# d: 154
pca = PCA(n_components = 154)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
print(x_train.shape)
print(y_train.shape)


# 2. 모델
parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},                                # 4번
    {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},               # 6번
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}      # 8번
]
kfold = KFold(n_splits = 5, shuffle = True)

model = GridSearchCV(XGBClassifier(n_jobs = 8, n_estimators = 1, use_label_encoder = False), 
                                   parameters, cv = kfold)
model.fit(x_train, y_train , eval_metric = 'mlogloss', verbose = True, 
          eval_set = [(x_train, y_train), (x_test, y_test)])

result = model.score(x_test, y_test)
print('Best Parameter: ', model.best_estimator_)
print('Acc: ', result)
