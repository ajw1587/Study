from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

print('__________')

x, y = load_boston(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, shuffle = True, random_state = 66
)

model = XGBRegressor(n_jobs = 8)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('R2: ', score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold = thresh, prefit = True)
    # threshold 값보다 높은 중요도들을 추출
    print(selection)

    select_x_train = selection.transform(x_train)
    # x_train의 feature를 threshold 기준으로 줄여준다.
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print('Thresh = %.3f, n = %d, R2: %.2f%%' %(thresh, select_x_train.shape[1],
          score*100))

# print(model.coef_)
# print(model.intercept_)
# AttributeError: Coefficients are not defined for Booster type None