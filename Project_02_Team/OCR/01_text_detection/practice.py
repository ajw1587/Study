import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 'F:/Team Project/Image_data/01.jpg'
# 'F:/Team Project/Image_data/0011.jpg'
# 'F:/Team Project/Image_data/0012.jpg'
# 'F:/Team Project/Image_data/0013.jpg'

# 'F:/Team Project/Image_data/02.jpg'
# 'F:/Team Project/Image_data/03.png'
# 'F:/Team Project/Image_data/ex04.png'
img = cv.imread('F:/Team Project/OCR/Text_detection/Image_data/0011.jpg', cv.IMREAD_COLOR)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2 = cv.bitwise_not(img2)
img2 = np.where(img2 < 50, 0, 255)
img2 = img2/255.
# pixel값 줄여주기 = 겹치는 글자를 최소화 해주기 위해
kernel = np.ones((2, 2), np.uint8)
img2 = cv.erode(img2, kernel, iterations=1)
# print(img2.shape)   # (2400, 1080)
# print(img2)
# cv.imshow('img', img2)
# cv.waitKey(0)
# cv.destroyAllWindows()


# Line Histogram
def Line_Histogram(gray_img):
    line_sum = gray_img.sum(axis=1)
    # print(line_sum)
    hist_label = np.arange(0, gray_img.shape[0])
    # print(line_sum.shape)
    # print(hist_label.shape)

    plt.figure(figsize=(10, 10))
    plt.barh(hist_label, line_sum)
    plt.show()

# Word Histogram
def Word_Histogram(Line_img):
    word_sum = Line_img.sum(axis=0)
    word_label = np.arange(0, Line_img.shape[1])
    # print(np.max(word_sum))
    # print(word_sum.shape)
    # print(word_label.shape)

    plt.figure(figsize=(10, 5))
    plt.bar(word_label, word_sum)
    plt.show()

# Line Split
def Line_Split(gray_img):
    line_sum = gray_img.sum(axis=1)
    # print(line_sum)
    line_idx = []
    sign = False        # True: 추출중, False: 0인 지점
    for i in range(line_sum.shape[0]):
        # print(i)
        # print(line_sum.shape[0])
        if sign == False:
            if line_sum[i] == 0:
                continue
            line_idx.append(i)
            sign = True
        else:   # sign == True
            if i == line_sum.shape[0]-1:
                line_idx.append(i -1)
                break
            if line_sum[i] != 0:
                continue
            line_idx.append(i-1)
            sign = False

    # print('line_idx', line_idx)
    line_img = []
    for k in range(0, len(line_idx), 2):
        line_img.append(img2[line_idx[k]: line_idx[k + 1], :])

    return line_img, line_idx

# Word Split
def Word_Split(line_img, line_idx): # line_img: numpy
    word_sum = line_img.sum(axis=0)
    word_idx = []
    # print(type(line_idx))
    # print(line_idx)
    sign = False        # True: 추출중, False: 0인 지점
    for i in range(word_sum.shape[0]):
        if sign == False:
            if word_sum[i] == 0:
                continue
            word_idx.append(i)
            sign = True
        else:   # sign == True
            if word_sum[i] != 0:
                continue
            word_idx.append(i)
            sign = False

    # 1. 필요없는 작은 문자 지워주기: 왜 생기는지는 모르겠다. 위 erode로 인해 생기는것 같다.
    del_list = []
    for i in range(0, len(word_idx), 2):
        diff = word_idx[i+1] - word_idx[i]
        if diff == 1:
            del_list.append(word_idx[i])
            del_list.append(word_idx[i + 1])
    # print(del_list)
    for j in range(0, len(del_list), 2):
        word_idx.remove(del_list[j])
        word_idx.remove(del_list[j + 1])
    # print('word_idx', word_idx)

    word_img = []
    # print(word_idx)
    # print(len(word_idx))
    for k in range(0, len(word_idx), 2):
        word_img.append(img2[line_idx[0]: line_idx[1], word_idx[k]: word_idx[k + 1]])
        # cv.imshow('image', img2[line_idx[0] : line_idx[1] , word_idx[k]: word_idx[k + 1]])
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    # Line_Split
    line_sum = word_img[0].sum(axis=1)
    print(line_sum)
    cv.imshow('image', word_img[0])
    cv.waitKey(0)
    cv.destroyAllWindows()
    # print('word_img[0]', word_img[0])
    print(word_img[0])
    word_line_idx = Line_Split(word_img[0])

    return word_img

# 적용
# Line_Histogram(img2)
line_img, line_idx = Line_Split(img2)

for i in range(len(line_img)):
    word_img = Word_Split(line_img[i], line_idx[i * 2 : i * 2 + 2])
