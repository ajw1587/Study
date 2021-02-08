# 가중치 저장할것
# 1. model.save() 쓸것
# 2. pickle 쓸것

# 61 카피
# model.cv_results를 붙여서 완성

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn = build_model, verbose = 1)

def create_hyperparameters():
    batches = [50]
    optimizers = ['adam']
    dropout = [0.3]
    epochs = [1]
    return {'batch_size': batches, 'optimizer': optimizers,
             'drop': dropout, 'epochs': epochs}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)

# keras가 machine learning보다 후에 나왔기 때문에
# ml에서 keras가 적용되기 위해서는 위와 같이 wrapper가 필요하다.
search.fit(x_train, y_train, verbose = 1)
print('(1): ', search.best_params_)         # {'optimizer': 'rmsprop', 'drop': 0.1, 'batch_size': 30}
print('(2): ', search.best_estimator_)      # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000020089D9AB50>
print('(3): ', search.best_score_)          # 0.9459499915440878

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)         # 최종 스코어:  0.9599000215530396

# 저장
import pickle
# pickle.dump(search.best_estimator_, open('../data/modelcheckpoint/Keras64_Search.dat', 'wb'))
search.best_estimator_.model.save('../data/modelcheckpoint/Keras64_Search.h5')
# 출처: https://github.com/keras-team/keras/issues/4274

# 불러오기
from sklearn.metrics import accuracy_score
model2 = load_model('../data/modelcheckpoint/Keras64_Search.h5')
y_pred = model2.predict(x_test)

y_test = np.argmax(y_test, axis = 1)
y_pred = np.argmax(y_pred, axis = 1)
print(y_test.shape)
print(y_pred.shape)

acc = accuracy_score(y_test, y_pred)
print(acc)
