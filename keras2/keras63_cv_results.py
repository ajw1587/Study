# 61 카피
# model.cv_results를 붙여서 완성

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
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

score = pd.DataFrame(search.cv_results_)    # rank_test_score가 성능의 순서이다.
print(score[['params','mean_test_score', 'rank_test_score', 'split0_test_score','split1_test_score', 'split2_test_score']])
# print(score)
#    mean_fit_time  std_fit_time  mean_score_time  ...  mean_test_score std_test_score rank_test_score
# 0       1.317706      0.107403         0.455574  ...         0.939617       0.002597               1
