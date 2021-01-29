import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  # 이름만 회귀이고 분류모델이다
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)


# 2. 모델, pipeline: 모델 + 전처리

# model = Pipeline([('inwoo', MinMaxScaler()), ('ggong', SVC())])

preprocessing = [MinMaxScaler(), StandardScaler()]
for i in range(len(preprocessing)):
    model = make_pipeline(preprocessing[i], SVC())    # 위의 pipeline과 결과값 동일

    model.fit(x_train, y_train)

    results = model.score(x_test, y_test)
    print(preprocessing[i], '의 결과', np.round(results, 4))

# MinMaxScaler() 의 결과 0.8333
# StandardScaler() 의 결과 0.8667