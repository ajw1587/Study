# RandomForestClassifier
from sklearn.datasets


# 2. Model
parameters = [
    {'n_estimators': [100, 200]},
    {'max_depth': [6, 8, 10, 12]},
    {'min_samples_leaf': [3, 5, 6, 10], 'n_estimators': [100, 200], 'max_depth': [6, 8, 10, 12],\
        'min_samples_split': [2, 3, 5, 10], 'n_jobs': [-1, 2, 4]},
    {'min_samples_split': [2, 3, 5, 10]},
    {'n_jobs': [-1, 2, 4]}
]