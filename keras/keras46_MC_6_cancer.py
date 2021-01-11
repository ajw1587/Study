import numpy as np
from sklearn.datasets import load_breast_cancer
# 1. 데이터
datasets = load_breast_cancer()

# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape)      # (569, 30)
# print(y.shape)      # (569,)
# print(x[:5])
# print(y)

# 2. 전처리
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
file_path = './modelCheckpoint/k46_6_cancer_{epoch:02d}_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience = 3, mode = 'auto')
cp = ModelCheckpoint(filepath = file_path, monitor = 'val_loss', save_best_only = True, mode = 'auto')

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ['acc'])
model.fit(x_train, y_train, epochs = 150, validation_data = (x_val, y_val), batch_size = 6, callbacks = [es, cp])
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

# 3. 세번째 방법
# numpy.where 사용하기
# np.where(x<1, 0, 1) x가 1보다 작으면 0, 크면 1



# loss:  0.05254744738340378
# acc:  0.9912280440330505
# [[2.8004186e-13]
#  [8.5227144e-01]
#  [5.7899669e-02]
#  [9.4692174e-16]]