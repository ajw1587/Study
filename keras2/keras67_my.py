# 나를 찍어서 내가 남자인지 여자인지에 대해
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# 1. ImageDataGenerator 설정
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    height_shift_range = 0.2,
    width_shift_range= 0.2
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

# 2. flow_from_directory 설정
batch = 32
size = 64
train_flow = train_datagen.flow_from_directory(
    '../data/image/gender/train',
    target_size = (size, size),
    batch_size = batch,
    class_mode = 'binary'   
)

test_flow = test_datagen.flow_from_directory(
    '../data/image/gender/train',
    target_size = (size, size),
    batch_size = batch,
    class_mode = 'binary'
)

val_flow = test_datagen.flow_from_directory(
    '../data/image/gender/train',
    target_size = (size, size),
    batch_size = batch,
    class_mode = 'binary'
)

# 3. 모델
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

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

modelpath = '../data/modelcheckpoint/keras67_my_gender.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor = 'loss', patience =20, mode = 'auto')

model = my_model()
model.fit_generator(train_flow, 
                    steps_per_epoch = 1736//batch, 
                    epochs = 100,
                    validation_data = test_flow, 
                    validation_steps = 5,
                    callbacks = [es, cp])

loss, acc = model.evaluate_generator(test_flow, verbose = 1)
print('loss: ', loss)
print('acc: ', acc)

# loss:  0.1495472490787506
# acc:  0.9464285969734192


# predict 이미지 불러오기
import cv2
import numpy as np

me = cv2.imread('../data/image/inwoo.jpg')
print(type(me))     # <class 'numpy.ndarray'>
print(me.shape)     # (649, 427, 3)
me = cv2.resize(me, dsize = (64, 64), interpolation = cv2.INTER_LINEAR)
me = me.reshape(1, 64, 64, 3)

x_pred = test_datagen.flow(me)
y_pred = model.predict(x_pred)

print('남자일 확률: ', np.round((1-y_pred)*100, 2), '%')
print('여자일 확률: ', np.round(y_pred*100, 2), '%')
