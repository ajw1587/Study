# 실습
# cifar10 으로 vgg16 넣어서 만들것

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# vgg16 설정
vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3))
vgg16.trainable = False


# Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(x_train.shape)        # (50000, 32, 32, 3)
# print(y_train.shape)        # (50000, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape)        # (50000, 32, 32, 3)
print(y_train.shape)        # (50000, 10)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

# Model
initial_model = vgg16
last = initial_model.output
x = Flatten()(last)
x = Dense(32)(x)
x = Dense(16)(x)
output1 = Dense(10, activation = 'softmax')(x)
model = Model(inputs = initial_model.input, outputs = output1)
# model = Sequential()
# model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(10, activation = 'softmax'))

hist = model.compile(optimizer = Adam(learning_rate = 0.001),
                     loss = 'categorical_crossentropy',
                     metrics = ['acc'])
model.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_val, y_val))


result = model.evaluate(x_test, y_test, batch_size = 256)
y_predict = model.predict(x_test)

print("loss: ", result[0])
print("acc: ", result[1])
print("y_test[:10]: \n", y_test[:10])
print("y_predict[:10]: \n", y_predict[:10])