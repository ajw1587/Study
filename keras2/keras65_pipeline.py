# 61번을 파이프라인으로 구성!!!

import numpy as np
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

# Pipeline 설정
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
pipe = Pipeline([('scaler', MinMaxScaler()), ('model2', model2)])

# Hyperparameter 설정
def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return {'model2__batch_size': batches, 
            'model2__optimizer': optimizers,
            'model2__drop': dropout}
            # 이름을 붙여준 이유: 이름없이 대입하면 pipe자체 파라미터값으로 인식.
            # pipe에는 없는 파라미터들이므로 ERROR 발생
            # 그러므로, model2의 파라미터라는 것을 명시
            # 만약 pipe 파라미터를 조정하고 싶으면 model2__를 빼고 작성하면 된다.
hyperparameters = create_hyperparameters()

# RandomizedSearchCV에 pipe, hyperparameters 설정
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(pipe, hyperparameters, cv = 3)

# keras가 machine learning보다 후에 나왔기 때문에
# ml에서 keras가 적용되기 위해서는 위와 같이 wrapper가 필요하다.
search.fit(x_train, y_train)
print('(1): ', search.best_params_)   
print('(2): ', search.best_estimator_)
print('(3): ', search.best_score_)    

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)

# (1):  {'model2__optimizer': 'adam', 'model2__drop': 0.1, 'model2__batch_size': 10}
# (2):  Pipeline(steps=[('scaler', MinMaxScaler()),
#                 ('model2',
#                  <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001F3FEB827F0>)])
# (3):  0.9541999896367391
# 1000/1000 [==============================] - 1s 840us/step - loss: 0.1143 - acc: 0.9655
# 최종 스코어:  0.965499997138977