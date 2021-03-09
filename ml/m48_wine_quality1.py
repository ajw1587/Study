import pandas as pd

# 1. Load Data
score = pd.read_csv('../data/csv/data-01-test-score.csv')
wine = pd.read_csv('../data/csv/winequality-white.csv', sep = ';', index_col = None, header = 0)

print(wine.head())
print(wine.shape)           # (4898, 12)
print(wine.describe())

x = wine.iloc[:, :11].values
y = wine.iloc[:, 11].values
print(x.shape, y.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, shuffle = True, train_size = 0.8
)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
print(x_train.shape, x_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# model = KNeighborsClassifier()          # SCORE:  0.5642857142857143
model = RandomForestClassifier()        # SCORE:  0.6714285714285714
# model = XGBClassifier()                 # SCORE:  0.6755102040816326
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('SCORE: ', score)
