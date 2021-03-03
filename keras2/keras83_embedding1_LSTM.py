from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화예요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로예요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요',
        '규현이가 잘생기긴 했어요']

# 긍정 1, 부정 0
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

# 크기 맞춰주기!
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding = 'pre', maxlen = 5)      # pre: 앞자리를 0으로, post: 뒷자리를 0으로
# , maxlen(최대길이) = 4, truncating(앞에서 자를지 뒤에서 자를지) = 'pre'
print(pad_x)
print(pad_x.shape)      # (13, 5)

print(np.unique(pad_x))
print(len(np.unique(pad_x)))
# [ 0  1  2  3  4  5  6  7  8  9 10 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27] 11이 maxlen으로 인해 잘렸다.
# 28

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

model = Sequential()
model.add(Embedding(input_dim = 28, output_dim = 11, input_length = 5)) # input_length는 column 수
# input_length: Flatten()시 output * input_length로 연산
# model.add(Embedding(28, 11))
model.add(LSTM(32))
model.add(Dense(1, activation = 'sigmoid'))

# model.summary()
# Param = input_dim * output_dim


'''
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
model.fit(pad_x, labels, epochs = 100)

acc = model.evaluate(pad_x, labels)[1]
print(acc)
'''