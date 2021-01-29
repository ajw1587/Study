# 실습!!
# 피처임포턴스가 전체 중요도에서 0인 컬럼들을 제거하여 데이터셋을 재 구성후
# DecisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
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
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size = 0.2, random_state = 77
)

# 2. 모델
model = GradientBoostingClassifier()

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
result = get_column_index(model, 3)
print(result)

# [3.64829893e-04 2.20360701e-02 1.01677616e-03 5.07045321e-04
#  1.04723964e-03 1.30601225e-03 1.83929011e-03 9.53177162e-02
#  8.78042691e-04 6.21553809e-04 5.02045397e-03 2.17676284e-04
#  4.40287499e-03 8.71013135e-03 8.45239469e-04 7.19905650e-04
#  2.77293325e-03 1.64868582e-02 2.42113002e-06 2.25606305e-03
#  6.85678990e-02 3.68205979e-02 4.30118166e-01 1.85723308e-01
#  3.18275804e-03 3.26560632e-03 9.34959530e-03 9.62225622e-02
#  3.80185502e-04 1.88480755e-07]
# acc:  0.956140350877193
# [22, 23, 27]
###############################################################################

# model.feature_importances
x_data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
x_data = x_data.iloc[:, [[x for x in result]]]
y_data = dataset.target

x_data = x_data.values

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x_data, y_data, test_size = 0.2, random_state = 77
)

# 2. 모델
model1 = RandomForestClassifier(max_depth = 4)

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
# [1.37248121e-05 2.15903964e-02 1.85742636e-03 5.35570118e-04
#  3.74430517e-03 8.20741810e-04 1.05414982e-03 9.32648806e-02
#  8.13910394e-04 5.03127150e-04 2.40985655e-03 7.25813501e-04
#  5.78292407e-03 9.81694726e-03 9.08254807e-04 1.29410457e-04
#  2.78141448e-05 1.64823922e-02 9.73481761e-05 2.22546865e-03
#  6.88038925e-02 3.79877247e-02 4.28951322e-01 1.86287588e-01
#  3.18275807e-03 3.13550781e-03 9.87517808e-03 9.83886687e-02
#  5.18387019e-04 6.45108094e-05]
# acc:  0.956140350877193
# [22, 23, 27]

# model.feature_importances 이용하여 column 축소
# [0.45944395 0.54055605]
# acc:  0.8596491228070176