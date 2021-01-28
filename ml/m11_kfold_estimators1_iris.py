from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

# 2. 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)
kfold = KFold(n_splits = 6, shuffle = True)

allAlgorithms = all_estimators(type_filter = 'classifier')
# sklearn에 들어있는 classifier 모델들이 들어있다.

for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        
        scores = cross_val_score(model, x_train, y_train, cv = kfold)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 Accuracy: \n', scores)
    except:
        print(name, '은 없는 놈!')
        # continue 그냥 넘기고 싶을때


import sklearn
print(sklearn.__version__)      # 0.23.2
