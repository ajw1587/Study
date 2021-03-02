import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# File Path
train_filepath = '../data/csv/Computer_Vision/data/train.csv'
test_filepath = '../data/csv/Computer_Vision/data/test.csv'
submission_filepath = '../data/csv/Computer_Vision/data/submission.csv'
check_filepath = '../data/modelcheckpoint/Computer_Vision/Computer_Vision_best_model.h5'

# 1. 데이터 불러오기
train_dataset = pd.read_csv(train_filepath, engine = 'python', encoding = 'CP949')
test_dataset = pd.read_csv(test_filepath, engine = 'python', encoding = 'CP949')
submission_dataset = pd.read_csv(submission_filepath, engine = 'python', encoding = 'CP949')

# 2. 데이터 필요 column 추려주기
train_x = train_dataset.copy().drop(['id', 'letter', 'digit'], 1).values
train_y = train_dataset.copy().loc[:,'digit'].values
test_x = test_dataset.copy().drop(['id', 'letter'], 1).values
submission_data = submission_dataset.copy()

train_x = train_x/255.
test_x = test_x/255.

# x_train, x_val, y_train,y_val = train_test_split(train_x, train_y, train_size = 0.8, random_state = 77)

x_train = train_x.reshape(train_x.shape[0], 28, 28, 1)
# x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = test_x.reshape(test_x.shape[0], 28, 28, 1)


# 3. 모델 정의
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input
from tensorflow.keras.layers import Dropout, BatchNormalization, Reshape, LeakyReLU

######################################################## 3-1. Encoder
encoder_input = Input(shape=(28, 28, 1))

# 28 X 28
x = Conv2D(32, 3, padding='same')(encoder_input) 
x = BatchNormalization()(x)
x = LeakyReLU()(x) 

# 28 X 28 -> 14 X 14
x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x) 
x = LeakyReLU()(x) 

# 14 X 14 -> 7 X 7
x = Conv2D(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 17 X 7
x = Conv2D(64, 3, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

# 2D 좌표로 표기하기 위하여 2를 출력값으로 지정합니다.
encoder_output = Dense(2)(x)

encoder = Model(encoder_input, encoder_output)

######################################################## 3-2. Decoder
# Input으로는 2D 좌표가 들어갑니다.
decoder_input = Input(shape=(2, ))

# 2D 좌표를 7*7*64 개의 neuron 출력 값을 가지도록 변경합니다.
x = Dense(7*7*64)(decoder_input)
x = Reshape((7, 7, 64))(x)

# 7 X 7 -> 7 X 7
x = Conv2DTranspose(64, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 7 X 7 -> 14 X 14
x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 14 X 14 -> 28 X 28
x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 28 X 28 -> 28 X 28
x = Conv2DTranspose(28, 3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 최종 output
decoder_output = Conv2DTranspose(1, 3, strides=1, padding='same', activation='tanh')(x)

decoder = Model(decoder_input, decoder_output)
decoder.summary()
print(x_train.shape)
print(x_test.shape)

# Hyper Parameter
LEARNING_RATE = 0.0005
BATCH_SIZE = 32

# encoder, decoder 연결
encoder_in = Input(shape=(28, 28, 1))
x = encoder(encoder_in)
decoder_out = decoder(x)

# AutoEncoder 최종 정의
auto_encoder = Model(encoder_in, decoder_out)
auto_encoder.compile(optimizer = Adam(LEARNING_RATE), loss = 'mse')
checkpoint_path = '/content/drive/My Drive/AIStudy/Dacon_Computer_Vision/Dacon_Computer_Vision_AutoEncoder.h5'
checkpoint = ModelCheckpoint(checkpoint_path,
                             save_best_only = True,
                             save_weights_only = True,
                             monitor = 'loss',
                             verbose = 1)
auto_encoder.fit(x_train, x_train, 
                 batch_size=BATCH_SIZE, 
                 epochs=100
                #  callbacks=[checkpoint]
                #  validation_split = 0.2
                )
y_pred = auto_encoder.predict(x_test)
submission_dataset['digit'] = np.argmax(y_pred, axis = 1)
submission_dataset.to_csv(save_submissionpath, index = False)