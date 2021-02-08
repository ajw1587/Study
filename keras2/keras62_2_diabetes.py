from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input

# 1. 데이터
dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    train_size = 0.8,
                                                    random_state = 77)

# 2. 전처리
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print('#########################################################################################')
# 3. 모델
def diabetes_model(act = 'relu', drop = 0.3, node1 = 32, node2 = 64, node3 = 128,
                   opti = 'adam'):
    input = Input(shape = (x_train.shape[1], x_train.shape[2]))
    x = LSTM(node1, activation = act)(input)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    output = Dense(1, activation = act)(x)
    model = Model(inputs = input, outputs = output)

    model.compile(optimizer = opti, loss = 'mse')

    return model

def hyper_parameter():
    act = ['relu']
    drop = [0.2, 0.3]
    node1 = [16, 32]
    node2 = [32, 64]
    node3 = [128]
    validation = [0.2]
    epochs = [50]
    batch_size = [8, 16]
    return {'act': act,
            'drop': drop,
            'node1': node1,
            'node2': node2,
            'node3': node3,
            'validation_split': validation,
            'epochs': epochs,
            'batch_size': batch_size}
hyperparameter = hyper_parameter()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn = diabetes_model, verbose = 1)

search = RandomizedSearchCV(model, hyperparameter, cv = 3)
search.fit(x_train, y_train, verbose = 1)

print('best_parameter: \n', search.best_params_)
print('best_estimator: \n', search.best_estimator_)
print('best_score: \n', search.best_score_)

result = search.score(x_test, y_test)
print('result: ', result)

# best_parameter:
#  {'validation_split': 0.2, 'node3': 128, 'node2': 32, 'node1': 32, 'epochs': 50, 'drop': 0.2, 'batch_size': 8, 'act': 'relu'}
# best_estimator:
#  <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x0000026CA098D8E0>
# best_score:
#  -4809.187825520833
# result:  -4405.34814453125