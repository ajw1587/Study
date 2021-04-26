#. DATA정보정리
# pclass : passenger Class 승객등급
# survuved : 생존여부 (생존1, 사망0)
# name : 이름
# sex : 성별
# sibsp : 동승한 형제 또는 배우자 수
# parch : 동승한 보모 또는 자녀 수
# ticket : 티켓 번호
# fare : 승객 지불 요금
# cabin : 선실 이름
# embarked : 승선항(c = 쉘부르크, Q= 퀸즈타운, S=사우스 햄튼)
# body : 사망자 확인 번호
# bome.dest : 고향/목적지

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

train = 'F:\Personal Project\Kaggle\Dataset\Tabular Playground Series - Apr 2021/train.csv'
test = 'F:\Personal Project\Kaggle\Dataset\Tabular Playground Series - Apr 2021/test.csv'
submission = 'F:\Personal Project\Kaggle\Dataset\Tabular Playground Series - Apr 2021/sample_submission.csv'

cat_types={}

def process_data(filename):
    """
    first run sets cat_types
    """
    dd=pd.read_csv(filename).set_index("PassengerId")
    
    dd["Ticket2"]=dd["Ticket"].str[:2]
    dd["Cabin2"]=dd["Cabin"].str[:2]
    dd["Ticket_nums"]=dd["Ticket"].str.extract("([0-9]{3,})").astype(float)
    
    for cat_col in ["Sex", "Embarked", "Ticket2", "Cabin2"]:
        if cat_col not in cat_types:
            dd[cat_col]=dd[cat_col].astype("category")
            cat_types[cat_col]=dd[cat_col].cat.categories
        else:
            dd[cat_col]=dd[cat_col].astype("category").cat.set_categories(cat_types[cat_col])
    
    return dd

D=process_data(train)
D_test=process_data(test)




feats=["Pclass", "Sex", "SibSp", "Parch", "Embarked", "Fare", "Ticket2", "Cabin2", "Ticket_nums"]

X=D[feats]
y=D["Survived"]

model = LGBMClassifier(
    learning_rate=0.01,
    num_leaves=20,
    n_estimators=300,
)

model.fit(X, y)

X_test = D_test[feats]

pred = model.predict(X_test)

submission=pd.DataFrame({"PassengerId": X_test.index, "Survived": pred})
submission.to_csv('F:\Personal Project\Kaggle\Dataset\Tabular Playground Series - Apr 2021/lgbm_result_submission.csv', index=False)
submission.head()