import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math


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
        if sign == False:
            if line_sum[i] == 0:
                continue
            line_idx.append(i)
            sign = True
        else:   # sign == True
            if i == line_sum.shape[0]-1:    # 마지막 pixel이 0이 아닐때 생기는 오류 방지
                line_idx.append(i -1)
                break
            if line_sum[i] != 0:
                continue
            line_idx.append(i)
            sign = False

    # print(line_idx)
    line_img = []
    for k in range(0, len(line_idx), 2):
        line_img.append(img2[line_idx[k]: line_idx[k + 1], :])

    return line_img, line_idx

# Word Split
def Word_Split(line_img, line_idx, num): # line_img: numpy
    word_sum = line_img.sum(axis=0)
    word_idx = []

    Word_Histogram(line_img)
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

    # 1. 필요없는 작은 문자(1 Pixel) 지워주기: 왜 생기는지는 모르겠다. 위 erode로 인해 생기는것 같다.
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
    # print(word_idx)



    # 2. '가' -> 'ㄱ' 'ㅏ'로 나오는 현상 없애주기: 각 글자의 중심값을 계산하여 이어주기
    # 글자 하나하나의 중심값 찾아주기
    # line_idx: 하나의 line
    # word_idx: 한 line내 글자들의 위치
    # img2[line_idx[0]: line_idx[1], word_idx[i]: word_idx[i + 1]]

    # 글자 이미지 잘라주기 -> 위의 잘라준 글자 이미지는 높이가 똑같다. 그래서 글자 자체의 높이를 구한다.
    wline_idx = []
    for i in range(0, len(word_idx), 2):
        wline_sum = img2[line_idx[0]: line_idx[1], word_idx[i]: word_idx[i + 1]].sum(axis = 1)
        box = []
        sign = False        # True: 추출중, False: 0인 지점
        for i in range(wline_sum.shape[0]):
            if sign == False:
                if wline_sum[i] == 0:
                    continue
                box.append(i)
                sign = True
            else:   # sign == True
                if i == wline_sum.shape[0]-1:    # 마지막 pixel이 0이 아닐때 생기는 오류 방지
                    box.append(i -1)
                    break
                if wline_sum[i] != 0:
                    continue
                box.append(i)
                sign = False
        wline_idx.append(box[0])
        wline_idx.append(box[-1])
    # print(wline_idx)

    # 이미지의 중심값
    text_center_loc = []
    for i in range(0, len(word_idx), 2):
        x_center_loc = word_idx[i] + (word_idx[i + 1] - word_idx[i]) / 2
        y_center_loc = wline_idx[i] + (wline_idx[i] - wline_idx[i + 1]) / 2
        text_center_loc.append(x_center_loc)
        text_center_loc.append(y_center_loc)
    
    # Text끼리의 중심값 거리
    distance_list = []
    dis_avg = 0
    for i in range(0, len(word_idx) - 2, 2):
        distance = math.sqrt((text_center_loc[i + 2] - text_center_loc[i])**2 + (text_center_loc[i + 3] - text_center_loc[i + 1])**2)
        distance_list.append(distance)
        dis_avg = sum(distance_list)/len(distance_list)
    print(distance_list)
    # print(np.array(text_center_loc).shape)
    # print(np.array(distance_list).shape)
    print(dis_avg * 0.65)

    # 2. '가' -> 'ㄱ' 'ㅏ'로 나오는 현상 없애주기: 각 글자의 중심값을 계산하여 이어주기
    center_list = []
    del_list = []
    sign = True
    for i in range(0, len(word_idx) - 2, 2):
        # a = (word_idx[i + 1] + word_idx[i])/2
        # b = (word_idx[i + 3] + word_idx[i + 2])/2
        # # print(a)
        # # print(b)
        # c_subtract = b - a
        # center_list.append(c_subtract)
        distance = math.sqrt((text_center_loc[i + 2] - text_center_loc[i])**2 + (text_center_loc[i + 3] - text_center_loc[i + 1])**2)
        if distance <= (dis_avg*0.2):
            continue
        if distance <= (dis_avg*0.68) and sign == True:    # and c_subtract > 25
            del_list.append(word_idx[i + 1])
            del_list.append(word_idx[i + 2])
            sign = False
        else:
            sign = True


    print(np.array(word_idx).shape)
    for j in range(0, len(del_list), 2):
        word_idx.remove(del_list[j])
        word_idx.remove(del_list[j + 1])



    # 확인
    # for i in range(0, len(word_idx), 2):
    #     cv.imshow('test', line_img[wline_idx[i]: wline_idx[i + 1], word_idx[i]: word_idx[i + 1]])
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()

    word_img = []
    for k in range(0, len(word_idx), 2):
        word_img.append(img[line_idx[0]: line_idx[1], word_idx[k]: word_idx[k + 1]])
        # cv.imwrite('F:/Team Project/Image_data/test_picture/test' + str(num) + '_' + str(k) + '.png', img[line_idx[0]: line_idx[1], word_idx[k]: word_idx[k + 1]])
        cv.imshow('image', img[line_idx[0] : line_idx[1] , word_idx[k]: word_idx[k + 1]])
        cv.waitKey(0)
        cv.destroyAllWindows()
    return word_img




# 적용
# 'F:/Team Project/OCR/Text_detection/Image_data/01.jpg'
# 'F:/Team Project/OCR/Text_detection/Image_data/0011.jpg'
# 'F:/Team Project/OCR/Text_detection/Image_data/0012.jpg'
# 'F:/Team Project/OCR/Text_detection/Image_data/0013.jpg'

# 'F:/Team Project/OCR/Text_detection/Image_data/02.jpg'
# 'F:/Team Project/OCR/Text_detection/Image_data/03.png'
# 'F:/Team Project/OCR/Text_detection/Image_data/ex04.png'
img = cv.imread('F:/Team Project/OCR/Text_detection/Image_data/0014.jpg', cv.IMREAD_COLOR)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img2 = cv.bitwise_not(img2)
img2 = np.where(img2 < 50, 0, 255)
img2 = img2/255.
# pixel값 줄여주기 = 겹치는 글자를 최소화 해주기 위해
kernel = np.ones((2, 2), np.uint8)
img2 = cv.erode(img2, kernel, iterations=1)
# print(img2.shape)   # (2400, 1080)
# print(img2)
cv.imshow('img', img2)
cv.waitKey(0)
cv.destroyAllWindows()

Line_Histogram(img2)
line_img, line_idx = Line_Split(img2)

# word_img = Word_Split(line_img[0], line_idx[0 : 2], 0)
for i in range(len(line_img)):
    word_img = Word_Split(line_img[i], line_idx[i * 2 : i * 2 + 2], i)
