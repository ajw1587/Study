import pandas as pd
import numpy as np


from pandas import read_csv
train = read_csv('../data/csv/Sunlight_generation/train/train.csv', index_col=None, header=0)
# train Data 불러오기
submission = read_csv('../data/csv/Sunlight_generation/sample_submission.csv', index_col=None, header=0)
# 완성파일 양식 불러오기

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    # 얕은 복사
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    # 필요 컬럼 추출

    if is_train==True:          
        # Train Data: 하루치로 그 다음 2일치를 예상
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        # 새로운 컬럼 추가 및 shift로 데이터 끌어올리기, 끌어올린 후 결측치 ffill 처리
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill')
        # 새로운 컬럼 추가 및 shift로 데이터 끌어올리기, 끌어올린 후 결측치 ffill 처리
        temp = temp.dropna()
        # 결측치 제거
        return temp.iloc[:-96]
        # Target 값을 끌어 올린만큼 버리기
    elif is_train==False:
        # Test Data: Train과 같이 2일치를 예상
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        # 필요 컬럼 추출
        return temp.iloc[-48:, :]
        # Train과 맞춰주기, 하루치로 다음 2일치를 예상


df_train = preprocess_data(train)

df_test = []

for i in range(81):
    file_path = '../data/csv/Sunlight_generation/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    # Train과 맞춰주기, 하루치로 다음 2일치를 예상
    df_test.append(temp)
    # 0~80 test data 합치기

X_test = pd.concat(df_test)
print(X_test.shape)
print(X_test.head())

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=32)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=32)
# train, val 데이터 나누기

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# quantiles 가 대체 뭐지?!?
# 사전적 뜻: 분위수, box plot 할때 그 분위수인가?!?
print(X_train_1.shape)

from lightgbm import LGBMRegressor
# lightgbm: Gradient Boosting 프레워크로 Tree 기반 학습 알고리즘
#           10,000 이상의 row (행) 을 가진 데이터에 사용하는 것을 권유
#           100개 이상의 파라미터 보유
#           파라미터 소개: https://nurilee.com/2020/04/03/lightgbm-definition-parameter-tuning/

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                   
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)

    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model

# Target 예측

def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    
    return LGBM_models, LGBM_actual_pred

# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
results_1.sort_index()[:48]

# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)
results_2.sort_index()[:48]

results_1.sort_index().iloc[:48]
results_2.sort_index()

print(results_1.shape, results_2.shape)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values

submission.to_csv('../data/sample_submission.csv', index=False)
