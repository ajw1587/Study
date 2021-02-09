# 실습
# cifar10을 flow로 구성해서 완성
# ImageDataGenerator, fit_generator

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Input, BatchNormalization, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)        # (50000, 32, 32, 3)
print(y_train.shape)        # (50000, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# # 그래프로 확인하기
# fig, axs = plt.subplots(1,2,figsize=(15,5))
# # Count plot training set
# sns.countplot(data = y_train, ax=axs[0])
# axs[0].set_title('Distribution of training data')
# axs[0].set_xlabel('Classes')
# # Count plot testing set
# sns.countplot(data = y_test, ax=axs[1])
# axs[1].set_title('Distribution of Testing data')
# axs[1].set_xlabel('Classes')
# plt.show()
# y는 10개로 분류됨.

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, random_state = 77)

# 2. ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1./255.,
    width_shift_range = 0.2,
    height_shift_range = 0.2
)
else_datagen = ImageDataGenerator(
    rescale = 1./255
)

size = 32
train_generator = train_datagen.flow(x_train, y_train, batch_size = size)   #, class_mode = 'categorical'안쓴다
val_generator = else_datagen.flow(x_val, y_val, batch_size = size)
test_generator = else_datagen.flow(x_test, y_test, batch_size = size)

# 3. 모델
def my_model():
    input = Input(shape = (32, 32, 3))
    x = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, 2, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 2, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 2, 1, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(64, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation = 'relu')(x)
    x = BatchNormalization()(x)
    output = Dense(10, activation = 'softmax')(x)
    model = Model(inputs = input, outputs = output)

    model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = 'acc')

    return model

model = my_model()
model.fit_generator(train_generator, steps_per_epoch = len(x_train)/size, epochs = 100,
                    validation_data = val_generator, validation_steps = 32)
loss, acc = model.evaluate_generator(test_generator, verbose = 1)
print('loss: ', loss)
print('acc: ', acc)

# loss:  0.8292364478111267
# acc:  0.8180999755859375