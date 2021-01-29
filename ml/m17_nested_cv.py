import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 이름만 회귀이고 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


# 분류에서는 score가 accuracy_score로 잡힌다.
# 1. 데이터
# x, y = load_iris(return_X_y= True)        x, y를 나눠주는 또다른 방법

dataset = load_iris()
x = dataset.data
y = dataset.target
# print(y)            # 0, 1, 2

# 2. k-fold 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
kfold = KFold(n_splits = 5, shuffle = True)     # n_splits: 몇도막으로 나누겠다.
parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},                                # 4번
    {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},               # 6번
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}      # 8번
]       # SVC Model에 들어가있는 parameter 설정

# 3. 모델
# model = SVC()
model = GridSearchCV(SVC(), parameters, cv = kfold)     # SVC Model에 parameters를 적용하고 데이터에는 kfold 적용
score = cross_val_score(model, x_train, y_train, cv = kfold)
# GridSearchCV()에서 가장 좋은 parameter값을 가진 model을 반환
# cross_val_score()에서 가장 좋은 score값을 반환

print('교차검증점수: ', score)

'''
# 4. 훈련
model.fit(x_train, y_train)

# 5. 평가 예측
print('최적의 매개변수 : ', model.best_estimator_)       # (4+6+8) * 5 번 동안 제일 좋은 매개변수를 출력
y_pred = model.predict(x_test)                          # (4+6+8) * 5 중 가장 좋은 predict값을 출력
print('최종정답률: ', accuracy_score(y_test, y_pred))
# model.score 사용해도 괜찮다.
'''