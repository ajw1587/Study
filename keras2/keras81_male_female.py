# [실습]VGG16 으로 만들어보기
# 나를 찍어서 내가 남자인지 여자인지에 대해
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16

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
vgg16 = VGG16(include_top = False, weights = 'imagenet', input_shape = (size, size, 3))
vgg16.trainable = False

initial_model = vgg16
last = initial_model.output
x = Flatten()(last)
x = Dense(32, activation = 'relu')(x)
x = Dense(16, activation = 'relu')(x)
output1 = Dense(1, activation = 'sigmoid')(x)
model = Model(inputs = initial_model.input, outputs = output1)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelcheckpoint/keras81_my_gender.hdf5'
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor = 'loss', patience =20, mode = 'auto')

model.compile(optimizer = Adam(learning_rate = 0.001), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])
model.fit_generator(train_flow, 
                    steps_per_epoch = 1736//batch, 
                    epochs = 100,
                    validation_data = test_flow, 
                    validation_steps = 5)
                    # callbacks = [es, cp])

loss, acc = model.evaluate_generator(test_flow, verbose = 1)
print('loss: ', loss)
print('acc: ', acc)