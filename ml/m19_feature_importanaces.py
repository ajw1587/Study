from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_iris()
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

# [0.01669101 0.         0.03210925 0.95119975]
# acc:  0.9