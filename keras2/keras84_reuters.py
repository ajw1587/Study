from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 10000, test_split = 0.2
)
'''
print(x_train[0], type(x_train[0]))
print(y_train[0])
print(len(x_train[0]), len(x_train[11]))
print('===============================================')
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)
# [1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 109, 15, 17, 12]
# 3
# 87 59
# ===============================================
# (8982,) (2246,)
# (8982,) (2246,)

print('뉴스기사 최대길이: ', max(len(l) for l in x_train))    # 2376
# a = np.max([len(l) for l in x_train])
print('뉴스기사 평균길이: ', sum(map(len, x_train)) / len(x_train)) # 145.539857

# plt.hist([len(s) for s in x_train], bins = 50)
# plt.show()


# y 분포
unique_elements, counts_elements = np.unique(y_train, return_counts = True)
# unique: 중복값 제거, return_counts: 중복횟수 반환
print('y분포: ', dict(zip(unique_elements, counts_elements)))
# 참고자료: https://datamod.tistory.com/77
# 뉴스기사 최대길이:  2376
# 뉴스기사 평균길이:  145.5398574927633
# y분포:  {0: 55, 1: 432, 2: 74, 3: 3159, 4: 1949, 5: 17, 6: 48, 7: 16, 8: 139, 9: 101, 10: 124, 
#          11: 390, 12: 49, 13: 172, 14: 26, 15: 20, 16: 444, 17: 39, 18: 66, 19: 549, 20: 269, 
#          21: 100, 22: 15, 23: 41, 24: 62, 25: 92, 26: 24, 27: 15, 28: 48, 29: 19, 30: 45, 31: 39, 
#          32: 32, 33: 11, 34: 50, 35: 10, 36: 49, 37: 19, 38: 19, 39: 24, 40: 36, 41: 30, 42: 13, 
#          43: 21, 44: 12, 45: 18}

plt.hist(y_train, bins = 46)
plt.show()


# x의 단어들 분포
print('===================================================================')
word_to_index = reuters.get_word_index()
print(word_to_index)
print(type(word_to_index))
print('===================================================================')



# 키와 벨류를 교체
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value] = key

print(index_to_word)

# 키 벨류 교환후
print(index_to_word)
print(index_to_word[1])
print(len(index_to_word))
print(index_to_word[30979])

# x_train[0]
print(x_train[0])
print(' '.join([index_to_word[index] for index in x_train[0]]))
'''

x_train = pad_sequences(x_train, maxlen = 2376, padding = 'pre')
print(x_train.shape)