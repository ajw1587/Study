# 실습!!
# 피처임포턴스가 전체 중요도에서 0인 컬럼들을 제거하여 데이터셋을 재 구성후
# DecisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

def get_column_index(model):
    feature = model.feature_importances_
    feature_list = []
    for i in feature:
        feature_list.append(i)
    feature_list.sort(reverse = True)
 
    result = []
    for j in range(len(feature_list)-len(feature_list)//4):
        result.append(feature.tolist().index(feature_list[j]))
    return result


# 1. 데이터
dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size = 0.2, random_state = 77
)

# 2. 모델
model = RandomForestClassifier()

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
result = get_column_index(model)
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
model1 = RandomForestClassifier()

# 3. 훈련
model1.fit(x1_train, y1_train)

# 4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)

print('축소된 column: ', model1.feature_importances_)   # column 중요도
print('column 축소 acc: ', acc1)

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
# [0.06615571 0.01365071 0.05114811 0.0334154  0.00370729 0.01026399
#  0.03418539 0.09220649 0.00243098 0.00536286 0.01059878 0.00447309
#  0.01703351 0.03249176 0.00359532 0.00478109 0.00731446 0.00793939
#  0.00655743 0.0044168  0.08080581 0.01647871 0.17922637 0.10816092
#  0.00890765 0.02380858 0.03272204 0.12103322 0.00949528 0.00763285]
# acc:  0.9385964912280702
# [22, 27, 23, 7, 20, 0, 2, 6, 3, 26, 13, 25, 12, 21, 1, 10, 5, 28, 24, 17, 29, 16, 18]

# 축소된 column:  [0.08241375 0.10606444 0.11458679 0.10807665 0.12305372 0.09197949
#  0.0495458  0.07254541 0.0581384  0.03005352 0.02679664 0.01922435
#  0.01291626 0.02176339 0.0178751  0.01813866 0.0071953  0.01129795
#  0.00923007 0.00474708 0.00503811 0.00522107 0.00409806]
# column 축소 acc:  0.9473684210526315