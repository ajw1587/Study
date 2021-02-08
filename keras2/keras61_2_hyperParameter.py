# CNN으로 수정
# 파라미터 변경할것
# 필수: 노드의 개수


import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1. 데이터  / 전처리
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

# 2. 모델
def build_model(drop = 0.5, lr = 0.01, outputlayer = 10,
                hiddenlayer = 128, act_func = 'relu',
                batch_size = 32): # optimizer = 'adam', 
    inputs = Input(shape = (28, 28, 1), name = 'input')
    x = Conv2D(512, 2, 1, activation = act_func, name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation = act_func, name = 'hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(hiddenlayer, activation = act_func, name = 'hidden3')(inputs)
    x = Dropout(drop)(x)
    x = Flatten()(x)
    outputs = Dense(outputlayer, activation = 'softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.summary()
    model.compile(optimizer = Adam(learning_rate = lr), metrics = ['acc'],
                  loss = 'categorical_crossentropy')
    return model
model2 = build_model()

# RandomizedSearcvCV, GridSearcvCV에 적용하기 위한 wrapper
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn = build_model, verbose = 1)

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    # optimizers = ['rmsprop', 'adam', 'adadelta']
    lr = [0.1, 0.01, 0.001]
    dropout = [0.1, 0.2, 0.3]
    hiddenlayer = [32, 62, 128]
    act_func = ['relu', 'tanh']
    epoch = [20, 30]
    return {'batch_size': batches, 'lr': lr,
            'drop': dropout, 'hiddenlayer': hiddenlayer,
            'act_func': act_func,
            'epochs': epoch}
hyperparameters = create_hyperparameters()

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv = 3)

# keras가 machine learning보다 후에 나왔기 때문에
# ml에서 keras가 적용되기 위해서는 위와 같이 wrapper가 필요하다.
search.fit(x_train, y_train, verbose = 1)
print('(1): ', search.best_params_)         # {'lr': 0.001, 'hiddenlayer': 62, 'drop': 0.1, 'batch_size': 50}
print('(2): ', search.best_estimator_)      # <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001DC04F7DA90>
print('(3): ', search.best_score_)          # 0.9110166827837626

acc = search.score(x_test, y_test)
print('최종 스코어: ', acc)                  # 0.9185000061988831