# 다차원 Dense 모델
# (n, 32, 32, 3) -> (n, 32, 32, 3)

from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train.shape: ', x_train.shape)     # (50000. 32. 32. 3)
print('x_test.shape: ', x_test.shape)       # (10000. 32. 32. 3)
print('y_train.shape: ', y_train.shape)     # (50000. 1)
print('y_test.shape: ', y_test.shape)       # (10000. 1)

# OneHotEncoder
from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import OneHotEncoder
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout

model = Sequential()
model.add(Dense(64, input_shape = (32, 32, 3), activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.summary()

'''
# Compile, Fit
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, batch_size = 256, epochs = 50, validation_split = 0.5) #, callbacks = [es, cp, reduce_lr])

# 응용
# y_test 10개와 y_test 10개를 출력하시오.
result = model.evaluate(x_test, y_test, batch_size = 256)
y_predict = model.predict(x_test)

print("loss: ", result[0])
print("acc: ", result[1])
print("y_test[:10]: \n", y_test[:10])
print("y_predict[:10]: \n", y_predict[:10])
'''