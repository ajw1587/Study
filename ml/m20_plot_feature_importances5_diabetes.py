from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size = 0.2, random_state = 77
)

# 2. 모델
model = DecisionTreeClassifier(max_depth = 4)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)   # column 중요도
print('acc: ', acc)


import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align = 'center')
    # plt.barh: 가로막대 그래프를 그리는 함수
    # 첫번째: bar가 그려질 위치
    # 두번째: 각 bar에 대한 수치
    # 세번째: bar의 정렬위치 설정
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

# [0.         0.         0.01041461 0.         0.01445293 0.11055382
#  0.06920194 0.         0.         0.38242179 0.01625954 0.
#  0.39669537]
# acc:  0.9166666666666666