import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, BatchNormalization, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 1. Augmentation 설정
train_datagen = ImageDataGenerator(
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rescale = 1./255.
)
test_datagen = ImageDataGenerator(
    rescale = 1./255.
)

# 2. Train, Val, Test Data 설정
size = 64
batch = 32
train_flow = train_datagen.flow_from_directory(
    '../data/image/gender/train',
    target_size = (size, size),
    class_mode = 'binary',
    batch_size = batch
)
test_flow = test_datagen.flow_from_directory(
    '../data/image/gender/train',
    target_size = (size, size),
    class_mode = 'binary',
    batch_size = batch
)
val_flow = test_datagen.flow_from_directory(
    '../data/image/gender/train',
    target_size = (size, size),
    class_mode = 'binary',
    batch_size = batch
)

# 3. Model 설정
optimizer = Adam(learning_rate = 0.001)
def my_model(opti = optimizer):
    input = Input(shape = (size, size, 3))
    x = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation = 'relu')(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation = 'sigmoid')(x)

    model = Model(inputs = input, outputs = output)
    model.compile(loss = 'binary_crossentropy', optimizer = opti, metrics = ['acc'])

    return model

model = my_model()
es = EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'auto')
model.fit_generator(
    train_flow,
    steps_per_epoch = 1736//batch,
    epochs = 100,
    validation_data = val_flow,
    validation_steps = 30,
    callbacks = es
)
loss, acc = model.evaluate_generator(test_flow, verbose = 1)
print('loss: ', loss)
print('acc: ', acc)

# 예측 이미지 가져오기
img = cv2.imread('../data/image/ggong.jpg')
img = cv2.resize(img, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
img = img.reshape(1, 64, 64, 3)

x_pred = test_datagen.flow(img)
y_pred = model.predict(x_pred)

print('남자일 확률: ', np.round((1-y_pred)*100, 2), '%')
print('여자일 확률: ', np.round(y_pred*100, 2), '%')

# loss:  0.021645255386829376
# acc:  0.9936636090278625
# 남자일 확률:  [[0.22]] %
# 여자일 확률:  [[99.78]] %
# 꽁이는 남자인데...