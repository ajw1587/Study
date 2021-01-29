from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. 데이터
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size = 0.2,
                                                    random_state = 77)

# 2. 모델
model = DecisionTreeClassifier()

# 3. Fit
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

# 5. 그래프
import numpy as np
import matplotlib.pyplot as plt

def show_feature_importances(model):
    n_feature = dataset.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_feature), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_feature)
show_feature_importances(model)
plt.show()