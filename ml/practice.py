from sklearn.datasets import load_boston
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    dataset.data,
    dataset.target,
    train_size = 0.8,
    random_state = 77
)

# 2. 모델
parameters = [
    {'forest__n_estimators': [100, 200]},
    {'forest__max_depth': [6, 8, 10, 12]},
    {'forest__min_samples_leaf': [3, 5, 6, 10]},
    {'forest__min_samples_split': [2, 3, 5, 10]},
    {'forest__n_jobs': [-1, 2, 4]}
]
kfold = KFold(n_splits = 5, shuffle = True)

sub_model1 = GridSearchCV(RandomForestRegressor(), parameters, cv = kfold)      # train, test 교차검증
sub_model2 = GridSearchCV(sub_model1, parameters, cv = kfold)                   # train, val 교차검증
model = Pipeline([('scaler', MinMaxScaler()), ('forest', sub_model2)])