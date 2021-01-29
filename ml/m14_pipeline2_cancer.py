# RandomForest로 구성할것
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

# 2. 모델
# model = Pipeline([('inwoo', MinMaxScaler()), ('ggong', RandomForestClassifier())])

preprocessing = [MinMaxScaler(), StandardScaler()]
for i in range(len(preprocessing)):
    model = make_pipeline(preprocessing[i], RandomForestClassifier())    # 위의 pipeline과 결과값 동일

    model.fit(x_train, y_train)

    results = model.score(x_test, y_test)
    print(preprocessing[i], '의 결과', np.round(results, 4))

# MinMaxScaler() 의 결과 0.9298
# StandardScaler() 의 결과 0.9474