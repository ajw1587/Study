# epochs: 100
# validation_split, callback 적용
# early_stopping 5 적용
# Reduce LR 3 적용
# modelcheckpoint 폴더에 hdf5 파일 저장

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

(x_train, y_train), (x_test, y_test) = mnist.load_data()
check_filepath = '../data/modelcheckpoint/keras61_4_{epoch:02d}_{val_loss:.4f}.hdf5'

# 1. 데이터  / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

# 2. 모델
def build_model(drop = 0.5, optimizer = 'adam'):
    inputs = Input(shape = (28*28,), name = 'input')
    x = Dense(512, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation = 'relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation = 'softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer = optimizer, metrics = ['acc'],
                  loss = 'categorical_crossentropy')

    return model
model2 = build_model()

# RandomizedSearcvCV, GridSearcvCV에 적용하기 위한 wrapper
# keras가 machine learning보다 후에 나왔기 때문에
# ml에서 keras가 적용되기 위해서는 위와 같이 wrapper가 필요하다.
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn = build_model, verbose = 1)

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['adam']
    dropout = [0.1, 0.2, 0.3]
    epochs = [2, 3]
    validation = [0.1, 0.2]
    es_num = 5
    lr_num = 3
    return {'batch_size': batches,
            'optimizer': optimizers,
            'drop': dropout, 
            'epochs': epochs, 
            'validation_split': validation,
            'es_num': es_num,
            'lr_num': lr_num}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)

es = EarlyStopping(monitor = 'val_loss', mode = 'auto')
# reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = lr_num)
# cp = ModelCheckpoint(check_filepath, monitor = 'val_loss', save_best_only = True, mode = 'auto')

search.fit(x_train, y_train, verbose = 1, callbacks = [es])

print('(1): ', search.best_params_)         # 
print('(2): ', search.best_estimator_)      # 
print('(3): ', search.best_score_)          # 

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)         # 최종 스코어: 
