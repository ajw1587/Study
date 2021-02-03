
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train.shape: ", x_train.shape)       # (60000, 28, 28)
print("y_train.shape: ", y_train.shape)       # (60000,)
print("x_test.shape: ", x_test.shape)         # (10000, 28, 28)
print("y_test.shape: ", y_test.shape)         # (10000,)

print("x_train[0].shape", x_train[0].shape)   # (28, 28)
print("y_train: \n ", y_train[:10])
# plt.imshow(x_train[0], 'hot')
# plt.show()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.
# (x_test.reshape(x_test.shape[0], x_test.shaep[1], x_test.shaep[2], 1))

# OneHotEncoding
# 여러분이 하시오!
# from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters = 100, kernel_size = (2,2), strides = 1, padding = 'same', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(100, 2, padding = 'same'))         # padding = valid: 패딩 사용 안함
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))
model.summary()

#==================================================================================================
# Model Save
model.save('../data/h5/k51_1_model1.h5')
#==================================================================================================

# 실습!! 완성하시오!!!
# 지표는 acc
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k51_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience =3, mode = 'auto')
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')         # 좋은 부분을 check!, filepath = 좋은 부분을 파일로 생성

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, batch_size = 10, epochs = 3, validation_split = 0.2, callbacks = [es, cp])

#==================================================================================================
# Model Save
model.save('./model/k51_1_model2.h5')           # 여기서 선언하면 가중치도 같이 저장된다.
#==================================================================================================

# 응용
# y_test 10개와 y_test 10개를 출력하시오.
result = model.evaluate(x_test, y_test, batch_size = 32)
y_predict = model.predict(x_test)

print("loss: ", result[0])
print("acc: ", result[1])

# 시각화
# import matplotlib.pyplot as plt

# plt.figure(figsize = (10, 6))           # 면적 잡아주기

# plt.subplot(2, 1, 1)                    # 2행1열짜리 그래프중 1번째
# plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
# plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
# plt.grid()

# # plt.title('손실비용')
# plt.title('Cost Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc = 'upper right')



# plt.subplot(2, 1, 2)                    # 2행1열짜리 그래프중 2번째
# plt.plot(hist.history['accuracy'], marker = '.', c = 'red')
# plt.plot(hist.history['val_accuracy'], marker = '.', c = 'blue')
# plt.grid()

# # plt.title('정확도')
# plt.title('Accuray')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['accuracy', 'val_accuracy'])        # 위치 명시 하지 않을시 자동으로 위치 선정

# plt.show()