from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.utils import to_categorical

# 1. 데이터
dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,
                                                    dataset.target,
                                                    train_size = 0.8,
                                                    random_state = 77)

# 2. 전처리
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 3. 모델
def diabetes_model(act = 'relu', drop = 0.3, node1 = 32, node2 = 64, node3 = 128,
                   opti = 'adam'):
    input = Input(shape = (x_train.shape[1], ))
    x = Dense(node1, activation = act)(input)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    x = Dense(node2, activation = act)(x)
    x = Dropout(drop)(x)
    output = Dense(3, activation = 'softmax')(x)
    model = Model(inputs = input, outputs = output)

    model.compile(optimizer = opti, loss = 'categorical_crossentropy', metrics = ['acc'])

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

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn = diabetes_model, verbose = 1)

search = RandomizedSearchCV(model, hyperparameter, cv = 3)
search.fit(x_train, y_train, verbose = 1)

print('best_parameter: \n', search.best_params_)
print('best_estimator: \n', search.best_estimator_)
print('best_score: \n', search.best_score_)

result = search.score(x_test, y_test)
print('result: ', result)