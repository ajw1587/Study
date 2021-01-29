# 실습!!
# 피처임포턴스가 전체 중요도에서 0인 컬럼들을 제거하여 데이터셋을 재 구성후
# DecisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

def get_column_index(model, num):
    feature = model.feature_importances_
    feature_list = []
    for i in feature:
        feature_list.append(i)
    feature_list.sort(reverse = True)

    result = []
    for j in range(num):
        result.append(feature.tolist().index(feature_list[j]))
    return result


# 1. 데이터
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size = 0.2, random_state = 77
)

# 2. 모델
model = GradientBoostingRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)   # column 중요도
print('acc: ', acc)

# 5. 표 그리기
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
result = get_column_index(model, 5)
print(result)


###############################################################################

# model.feature_importances
x_data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
x_data = x_data.iloc[:, [x for x in result]]
y_data = dataset.target

x_data = x_data.values

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x_data, y_data, test_size = 0.2, random_state = 77
)

# 2. 모델
model1 = GradientBoostingRegressor()

# 3. 훈련
model1.fit(x1_train, y1_train)

# 4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)

print(model1.feature_importances_)   # column 중요도
print('acc: ', acc1)

def plot_feature_importances(model):
    n_features = x1_train.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align = 'center')
    # plt.barh: 가로막대 그래프를 그리는 함수
    # 첫번째: bar가 그려질 위치
    # 두번째: 각 bar에 대한 수치
    # 세번째: bar의 정렬위치 설정
    plt.yticks(np.arange(n_features), x_train.dtype.names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances(model1)
plt.show()


# 일반 columns들 이용
# [2.62419056e-02 4.26021714e-04 9.99405600e-04 9.98182621e-04
#  3.42082023e-02 2.92224386e-01 1.12270656e-02 8.88063262e-02
#  9.40087037e-03 9.92445589e-03 2.52830723e-02 7.75021258e-03
#  4.92509893e-01]
# acc:  0.8977605797081353
# [12, 5, 7, 4, 0]

# model.feature_importances 이용하여 column 축소
# [0.52472766 0.29010982 0.1068417  0.04608654 0.03223428]
# acc:  0.880190405427278