import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 2. 모델
model = Sequential()
model.add(LSTM(200, input_shape = (4, 1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()


# 모델 저장     /, \ 둘다 상관 없다.
model.save("./model/save_keras35.h5")       # '.': 현재폴더: Study
model.save(".//model//save_keras35.h5")     # '.': 현재폴더: Study
model.save(".\model\save_keras35.h5")       # '.': 현재폴더: Study, 단 \n 이 있으면 오류가 발생할 수도 있다. 그떄는 \\n이라고 하면 된다.
model.save(".\\model\\save_keras35.h5")     # '.': 현재폴더: Study
    