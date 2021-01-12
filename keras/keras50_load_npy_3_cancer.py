import numpy as np

x = np.load('../data/npy/cancer_x.npy')
y = np.load('../data/npy/cancer_y.npy')

print(x)
print(y)
print(x.shape, y.shape)

# 모델을 완성하시오.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 60)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 60)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# 3. 모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation = "relu", input_shape = (30,)))
model.add(Dense(100, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(400, activation = "relu"))
model.add(Dense(500, activation = "relu"))
model.add(Dense(400, activation = "relu"))
model.add(Dense(300, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(100, activation = "relu"))
model.add(Dense(10, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

# 4. Compile and Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
es = EarlyStopping(monitor = 'loss', patience = 10, mode = 'auto')
cp = ModelCheckpoint(filepath = '../data/modelcheckpoint', monitor = 'val_loss', save_best_only = True, mode = 'auto')
tb = TensorBoard(log_dir = '../data/graph', histogram_freq = 0, write_graph = True, write_images = True)

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['acc'])
model.fit(x_train, y_train, epochs = 150, validation_data = (x_val, y_val), batch_size = 6, callbacks = [es, cp, tb])
loss, acc = model.evaluate(x_test, y_test, batch_size = 6)
print("loss: ", loss)
print("acc: ", acc)

# 실습1. loss 0.985 이상
# 실습2. y_predict 출력
y_predict = model.predict(x_test[-5:-1])

# 1. 첫번째 방법
for i in range(len(y_predict)):
    print(i)
    if y_predict[i] >=0.5:
        y_predict[i] = 1
    else:
        y_predict[i] = 0
y_predict = np.transpose(y_predict)

# # 2. 두번째 방법
# y_pre = list(map(int, np.round(y_predict,0)))
# print(y_pre)
# print(y_test[-5:-1])