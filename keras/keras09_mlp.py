import numpy as np

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 열은 특징, 행은 속성값 즉, 행무시 열우선
print(x.shape)      # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> (10,) -> input_dim = 1
print(x.shape)      # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] -> (2, 10) -> input_dim = 10

# 1. [ [1, 2, 3], [4, 5, 6] ]                   = (2, 3) -> 특징 2개, 속성값 3개씩 즉 (3,2)로 표현해야한다.
# 2. [ [1, 2], [3, 4], [5, 6] ]                 = (3, 2) -> 특징 3개, 속성값 2개씩 즉 (2,3)로 표현해야한다.
# 3. [ [ [1, 2, 3], [4, 5, 6] ] ]               = (1, 2, 3)
# 4. [ [1, 2, 3, 4, 5, 6] ]                     = (1, 6)
# 5. [ [ [1, 2], [3, 4] ], [ [5, 6], [7, 8] ] ] = (2, 2, 2)
# 6. [ [1], [2], [3] ]                          = (3, 1)
# 7. [ [ [1], [2] ], [ [3], [4] ] ]             = (2, 2, 1)
# 위와 같은 상황을 위해 행과열을 바꿔주는 함수는 numpy에 있다.

# (2,10) -> (10,2)
# print(x.T)                  # 3차원: 0, 2차원: 1, 1차원: 2
# print(np.swapaxes(x,0,1))   # 3차원: 0, 2차원: 1, 1차원: 2
x = np.transpose(x)
print(x)      # 3차원: 0, 2차원: 1, 1차원: 2
print(x.shape)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim = 2))     # input_dim = 컬럼 갯수
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

# 컴파일
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x,y, epochs = 100, batch_size = 1, validation_split = 0.2)

# 평가, 예측
loss, mae = model.evaluate(x, y)
print("loss: ", loss)
print("mae: ", mae)

y_predict = model.predict(x)
print("y_predict: ", y_predict)

'''
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

r2 = r2_score(y_test, y_predict)

print("RMSE: ", RMSE(y_test, y_predict))
print("R2: ", r2)
'''