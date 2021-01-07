import numpy as np

def split_x(a, size):
    aaa = []
    for i in range(len(a) - size + 1):
        subset = a[i: (i+size)]
        aaa.append(subset)
    print(aaa)
    return np.array(aaa)

# 데이터
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

a = np.array(range(1, 11))
a = split_x(a, 5)
x = a[:, :4]
y = a[:, 4:]
print(x.shape)      # (6,4)
print(y.shape)      # (6,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)


# 모델
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = "loss", patience = 10, mode = "auto")
model = load_model('./model/save_keras35.h5')
# model.summary()
model.compile(loss = "mse", optimizer = "adam", metrics = ["mae"])
model.fit(x_train, y_train, batch_size = 6, epochs = 200, validation_data = (x_val, y_val), callbacks = early_stopping)


# Evaluate and Predict
from sklearn.metrics import r2_score, mean_squared_error

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
y_test_predict = model.predict(x_test)
print("y_test_predict: ", y_test_predict.shape)
print("y_test_predict: \n", y_test_predict)
print("y_test: \n", y_test)

loss = model.evaluate(x_test, y_test, batch_size = 10)
rmse = RMSE(y_test, y_test_predict)
R2 = r2_score(y_test, y_test_predict)

print("loss: ", loss)
print("RMSE: ", rmse)
print("R2_SCORE: ", R2)

x_predict = np.array([[7, 8, 9, 10], [8, 9, 10, 11]])
x_predict = x_predict.reshape(2, 4, 1)

result = model.predict(x_predict)
print("result: \n", result)