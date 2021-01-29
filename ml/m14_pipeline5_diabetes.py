# RandomForest로 구성할것
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 77)

# 2. 모델
# model = Pipeline([('inwoo', MinMaxScaler()), ('ggong', RandomForestRegressor())])
# model = make_pipeline(MinMaxScaler(), RandomForestRegressor())
model = make_pipeline(StandardScaler(), RandomForestRegressor())

# 3. Fit
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print(results)

# MinMaxSclaer(): 0.4921749820525253
# StandardScaler(): 0.48790684159053166