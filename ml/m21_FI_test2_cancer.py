# 실습!!
# 피처임포턴스가 0인 컬럼들을 제거하여 데이터셋을 재 구성후
# DecisionTree로 모델을 돌려서 acc 확인!!!

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터
dataset = load_breast_cancer()
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

# 5. 표 그리기
import matplotlib.pyplot as plt
import numpy as np
'''
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
'''
# [0.         0.05668567 0.         0.         0.         0.
#  0.         0.08330191 0.         0.         0.00958658 0.
#  0.         0.         0.         0.         0.         0.03271744
#  0.         0.         0.00267224 0.01501254 0.73729345 0.
#  0.00714001 0.         0.00723819 0.04835197 0.         0.        ]
# acc:  0.9210526315789473

###############################################################################

# model.feature_importances
x_data = pd.DataFrame(dataset.data, columns = dataset.feature_names)
x_data = x_data.iloc[:, [1, 7, 10, 17, 20, 21, 22, 24, 26, 27]]
y_data = dataset.target

x_data = x_data.values

x1_train, x1_test, y1_train, y1_test = train_test_split(
    x_data, y_data, test_size = 0.2, random_state = 77
)

# 2. 모델
model1 = DecisionTreeClassifier(max_depth = 4)

# 3. 훈련
model1.fit(x1_train, y1_train)

# 4. 평가, 예측
acc1 = model1.score(x1_test, y1_test)

print(model1.feature_importances_)   # column 중요도
print('acc: ', acc1)

plot_feature_importances_dataset(model1)
plt.show()


# 일반 columns들 이용
# [0.         0.02077621 0.         0.         0.         0.
#  0.         0.08330191 0.         0.         0.         0.
#  0.00958658 0.         0.         0.         0.         0.03271744
#  0.         0.         0.00991043 0.0393409  0.74887455 0.
#  0.00714001 0.         0.         0.04835197 0.         0.        ]
# acc:  0.9122807017543859

# model.feature_importances 이용하여 column 축소
# [0.04510457 0.08330191 0.00958658 0.03271744 0.00267224 0.01501254
#  0.73729345 0.00714001 0.01881929 0.04835197]
# acc:  0.9210526315789473