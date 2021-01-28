# 모델: RandomForestClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)
kfold = KFold(n_splits = 5, shuffle = True)

# 2. Model
parameters = [
    {'n_estimators': [100, 200]},
    {'max_depth': [6, 8, 10, 12]},
    {'min_samples_leaf': [3, 5, 6, 10], 'n_estimators': [100, 200], 'max_depth': [6, 8, 10, 12],\
        'min_samples_split': [2, 3, 5, 10], 'n_jobs': [-1, 2, 4]},
    {'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1, 2, 4]}
]
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold)

# 3. Fit
model.fit(x_train, y_train)

# 4. 예측, 평가
print('Best Parameters: ', model.best_estimator_)
# y_pred = model.predict(x_test)
# print('Accuracy: ', accuracy_score(y_test, y_pred))
print('Accuracy: ', model.score(x_test, y_test))

# Best Parameters:  RandomForestClassifier(max_depth=6, min_samples_leaf=10, min_samples_split=5, n_jobs=4)
# Accuracy:  0.8666666666666667

##################################################################################
# GridSearchCV 매개변수

# 1. n_estimators
# - 결정트리의 갯수를 지정
# - Default = 10
# - 무작정 트리 갯수를 늘리면 성능 좋아지는 것 대비 시간이 걸릴 수 있음

# 2. max_depth
# - 트리의 최대 깊이
# - default = None
# → 완벽하게 클래스 값이 결정될 때 까지 분할
# 또는 데이터 개수가 min_samples_split보다 작아질 때까지 분할
# - 깊이가 깊어지면 과적합될 수 있으므로 적절히 제어 필요

# 3. min_samples_leaf
# - 리프노드가 되기 위해 필요한 최소한의 샘플 데이터수
# - min_samples_split과 함께 과적합 제어 용도
# - 불균형 데이터의 경우 특정 클래스의 데이터가 극도로 작을 수 있으므로 작게 설정 필요

# 4. min_samples_split
# - 노드를 분할하기 위한 최소한의 샘플 데이터수
# → 과적합을 제어하는데 사용
# - Default = 2 → 작게 설정할 수록 분할 노드가 많아져 과적합 가능성 증가

# 5. n_jobs
# - 사용할 코어 수

# 출처: https://injo.tistory.com/30