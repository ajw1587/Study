import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters, imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data 불러오기
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 20000)
# print(x_train.shape)    # (25000,)
# print(x_test.shape)     # (25000,)
# print(y_train.shape)    # (25000,)
# print(y_test.shape)     # (25000,)

# Data 분석
y_train_value, y_train_count = np.unique(y_train, return_counts = True)
# print(y_train_value)    # [0 1]           이진 데이터
# print(y_train_count)    # [12500 12500]   동등하게 분할

# plt.figure(figsize = (10, 5))
# plt.hist([len(i) for i in x_train], bins = 50, range = (np.min([len(i) for i in x_train]), np.max([len(i) for i in x_train])))
# plt.grid()
# plt.show()

# Train Data 전처리
x_train = pad_sequences(x_train, maxlen = 100, padding = 'pre')
x_test = pad_sequences(x_test, maxlen = 100, padding = 'pre')
print(x_train.shape)    # (25000, 100)
print(x_test.shape)     # (25000, 100)

# Model 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, Input

input1 = Input(shape = (100,))
x = Embedding(input_dim = 20000, output_dim = 64, input_length = 100)(input1)
x = Flatten()(input1)
x = Dense(32, activation = 'relu')(x)
output1 = Dense(1, activation = 'sigmoid')(x)

model = Model(inputs = input1, outputs = output1)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 32, verbose = 1)

results = model.evaluate(x_test, y_test)
print('loss: ', results[0])
print('acc: ', results[1])