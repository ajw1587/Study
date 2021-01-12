
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1)/255.


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
model.save('../data/h5/k52_1_model1.h5')            # 모델만 저장
#==================================================================================================

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor = 'loss', patience =3, mode = 'auto')
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss',       # 좋은 부분을 check!, filepaht = 좋은 부분을 파일로 생성
                     save_best_only = True, mode = 'auto')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_train, y_train, batch_size = 10, epochs = 8, validation_split = 0.2, callbacks = [es, cp])

#==================================================================================================
# Model Save
model.save('../data/h5/k52_1_model2.h5')            # 모델 + 가중치(epochs 끝나는시점의 가중치)도 같이 저장된다.
model.save_weights('../data/h5/k52_1_weight.h5')    # 가중치만 저장(fit만 저장)
#==================================================================================================

# 응용
# y_test 10개와 y_test 10개를 출력하시오.
result = model.evaluate(x_test, y_test, batch_size = 32)
print("loss: ", result[0])
print("acc: ", result[1])