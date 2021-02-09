# 실습
# 남자 여자 구별
# ImageDataGenerator, fit_generator 사용
# female: 895장, 300 x 300
# male: 841장, 300 x 300

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# 1. 데이터
train_aug = ImageDataGenerator(
    rescale = 1./255.,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
)

test_aug = ImageDataGenerator(
    rescale = 1./255.
)

train_xy = train_aug.flow_from_directory(
    '../data/image/gender/train',
    target_size = (30, 30),
    batch_size = 14,
    class_mode = 'binary'
    # ,save_to_dir = '../data/image/gender_generator/train'
)

test_xy = test_aug.flow_from_directory(
    '../data/image/gender/train',
    target_size = (30, 30),
    batch_size = 32,
    class_mode = 'binary'
)

# 2. 모델
optimizer = Adam(learning_rate = 0.001)
def my_model(opti = optimizer):
    input = Input(shape = (30, 30, 3))
    x = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, 1, padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, 1, padding = 'same', activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
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
model.fit_generator(train_xy,
                    steps_per_epoch = 124,
                    epochs = 100,
                    validation_data = test_xy,
                    validation_steps = 2)

loss, acc = model.evaluate_generator(test_xy, verbose = 1)
print('loss: ', loss)
print('acc: ', acc)

# loss:  0.4948316514492035
# acc:  0.7413594722747803