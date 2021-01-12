import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data        # (178,13)
y = dataset.target      # (178,)

print("x: ", x)
print("x.shape: ", x.shape)
print("y: ", y)
print("y.shape: ", y.shape)

# 실습, DNN 완성할것
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 77)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


# 3. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(10, activation = "relu", input_shape = (13,)))
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(300, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(300, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(300, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(200, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation = "relu"))
model.add(Dense(3, activation = "softmax"))     # 다중분류의 경우 나누고자 하는 종류의 숫자를 기입하고 softmax를 사용한다.
                                                # 원핫인코딩, to_categorical -> wikidocs.net/22647


# 4. Compile and Train
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
file_path = '../data/modelcheckpoint/k46_8_wine_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['acc'])
model.fit(x_train, y_train, epochs = 200, validation_data = (x_val, y_val), batch_size = 3, callbacks = [es, cp])
loss, acc = model.evaluate(x_test, y_test)
y_test_predict = model.predict(x_test)
rmse = RMSE(y_test, y_test_predict)

print("loss: ", loss)
print("acc: ", acc)
print("RMSE: ", rmse)

y_predict = model.predict(x_test[-5:-1])
print(y_predict)
print(y_test[-5:-1])