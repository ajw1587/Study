# DICTIONARY 만들기
import numpy as np
import pandas as pd

LABEL_PATH = 'C:/Study/Project_02_Team/OCR/tensorflow-hangul-recognition-master/labels/2350-common-hangul-3.txt'

label_txt = open(LABEL_PATH, 'r', encoding = 'UTF8')
# UnicodeDecodeError: 'cp949' codec can't decode byte 0x80 in position 2: illegal multibyte sequence
# 에러 발생시 encoding = 'UTF8' 추가

label = label_txt.read()
print(type(label))
print('1: ', len(label))
print('2: ', label[0])
print('3: ', label[1])
print('4: ', label[2])

# txt 파일안의 ENTER(\n)값 제거해주기
label_list = []
for i in range(len(label)):
    if label[i] != '\n':
        label_list.append(label[i])

print(np.array(label_list).shape)
print('1: ', len(label_list))
print('2: ', label_list[0])
print('3: ', label_list[1])
print('4: ', label_list[2])

# DICTIONARY 만들어주기
label_dic = {}
for i in range(len(label_list)):
    label_dic[i] = label_list[i]

# 생성된 DICTIONARY를 PICKLE로 저장시켜주기
