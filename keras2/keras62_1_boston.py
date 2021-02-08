from sklearn.datasets import load_boston
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# 1. 데이터
dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    train_size = 0.8,
                                                    random_state = 77)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# 2. 모델
def LSTM_model(node1 = 32,
               node2 = 64,
               node3 = 128,
               drop = 0.2,
               act = 'relu',
               opti = 'adam'):
    input = Input(shape = (x_train.shape[1], x_train.shape[2]))
    x = LSTM(node1, activation = act)(input)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node3, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node1, activation = act)(x)
    x = Dropout(drop)(x)
    output = Dense(1, activation = act)(x)
    model = Model(inputs = input, outputs = output)
    model.compile(optimizer = opti,
                  loss = 'mse')
    return model
# model = LSTM_model()

# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# model = KerasRegressor(build_fn = LSTM_model, verbose = 1)

def hyper_parameters():
    dropout = [0.1, 0.2, 0.3]
    act = ['relu']
    opti = ['adam']
    node1 = [32, 64, 128]
    node2 = [32, 64, 128]
    node3 = [32, 64, 128]
    batches = [10, 20, 30, 40, 50]
    epochs = [100]
    validation = [0.1, 0.2]
    return {'drop': dropout, 
            'act': act,
            'opti': opti,
            'node1': node1,
            'node2': node2,
            'node3': node3,
            'batch_size': batches,
            'epochs': epochs,
            'validation_split': validation}
hyperparameters = hyper_parameters()

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
model = KerasRegressor(build_fn = LSTM_model, verbose = 1)

search = RandomizedSearchCV(model, hyperparameters, cv = 3)
search.fit(x_train, y_train, verbose = 1)

print('best_parameter: \n', search.best_params_)
print('best_estimator: \n', search.best_estimator_)
print('best_score: \n', search.best_score_)

result = search.score(x_test, y_test)
print('result: ', result)

# best_parameter: 
#  {'validation_split': 0.1, 'opti': 'adam', 'node3': 64, 'node2': 64, 'node1': 64, 'epochs': 100, 'drop': 0.1, 'batch_size': 20, 'act': 'relu'}
# best_estimator:
#  <tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor object at 0x000002090AE25670>
# best_score:
#  -43.3021437327067
# result:  -46.34390640258789