import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

def custom_mse(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
                                        # 알아서 인식해준다.
                                        # 첫번째는 true값, 두번쨰는 pred값
                                        # 이름은 상관없고 순서만 상관있다.
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

model.compile(loss = custom_mse, optimizer = 'adam')
                     # loss값은 함수로 정의해줄 수 있다.
model.fit(x, y, batch_size = 1, epochs = 50)

loss = model.evaluate(x, y)
print(loss)