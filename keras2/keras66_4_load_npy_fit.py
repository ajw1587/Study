import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, BatchNormalization, Dropout, Input, Flatten

x_train = np.load('../data/image/brain/npy/keras66_train_x.npy')
y_train = np.load('../data/image/brain/npy/keras66_train_y.npy')
x_test = np.load('../data/image/brain/npy/keras66_test_x.npy')
y_test = np.load('../data/image/brain/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# Found 160 images belonging to 2 classes.
# Found 120 images belonging to 2 classes.
# (160, 150, 150, 3) (160,)
# (120, 150, 150, 3) (120,)

# 실습
# 모델을 만들어랏!!!
input = Input(shape = (150, 150, 3))
x = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
x = BatchNormalization()(x)
x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)
x = Dropout(0.2)(x)

x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)
x = Dropout(0.2)(x)

x = Flatten()(x)
x = Dense(64, activation = 'relu')(x)
x = Dense(32, activation = 'relu')(x)
x = Dense(16, activation = 'relu')(x)
output = Dense(1, activation = 'sigmoid')(x)
model = Model(inputs = input, outputs = output)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])

history = model.fit(
    x_train, y_train, epochs = 80, batch_size = 8,
    validation_data = (x_test, y_test))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']