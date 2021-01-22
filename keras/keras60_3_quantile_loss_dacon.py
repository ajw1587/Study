import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K

def custom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

def quantile_loss(y_true, y_pred):
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q = tf.constant(np.array([qs]), dtype = tf.float32)        # 상수값으로 바꿔준다. 즉, 바뀌지 않는값
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def quantile_loss_dacon(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis = -1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 1. 데이터
x = np.array(range(1, 9)).astype('float32')
y = np.array(range(1, 9)).astype('float32')

print(x.shape)
print(y.shape)

# 2. 모델
model = Sequential()
model.add(Dense(10, input_shape = (1,)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(quantiles[0], y_true, y_pred), optimizer = 'adam')
             # loss값은 lambda로 정의해줄 수 있다.
model.fit(x, y, batch_size = 1, epochs = 50)

loss = model.evaluate(x, y)
print(loss)

# custom
# 0.08675681054592133

# quantile
# 0.0068542733788490295