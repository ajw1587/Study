from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# 1. 데이터
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data,
    dataset.target,
    train_size = 0.8,
    random_state = 77
)

# 2. 모델
model = DecisionTreeRegressor()

# 3. Fit
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)

# 5. 그래프
import matplotlib.pyplot as plt
import numpy as np

def show_feature_importances(model):
    n_feature = dataset.data.shape[1]
    plt.barh(np.arange(n_feature), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_feature), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_feature)
show_feature_importances(model)
plt.show()