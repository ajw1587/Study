# 실습
# cifar10 으로 inceptionV3 넣어서 만들것
# InceptionV3 는 75, 75만 가능하다!

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras. callbacks import EarlyStopping, ReduceLROnPlateau

# InceptionV3 설정
inceptionv3 = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3))
inceptionv3.trainable = False


# Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape)        # (50000, 32, 32, 3)
print(y_train.shape)        # (50000, 10)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

model = Sequential()
model.add(inceptionv3)
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(10, activation = 'softmax'))

hist = model.compile(optimizer = Adam(learning_rate = 0.001),
                     loss = 'categorical_crossentropy',
                     metrics = ['acc'])

es = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'auto')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, patience = 15)
model.fit(x_train, y_train, batch_size = 128, epochs = 100, validation_data = (x_val, y_val),
          callbacks = [es, reduce_lr])


result = model.evaluate(x_test, y_test, batch_size = 256)
y_predict = model.predict(x_test)

print("loss: ", result[0])
print("acc: ", result[1])