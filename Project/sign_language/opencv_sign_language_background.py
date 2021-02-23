import cv2
import numpy as np

def empty(a):
    pass

# 영상 및 이미지에서 손 검출하기
path = '../data/sign_image/my_hand2.mp4' # _image.png

cap = cv2.VideoCapture(path)

if cap.isOpened():
    print('Video Error')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

while True:
    ret, frame = cap.read()
    # img = cv2.imread(path)
    img = cv2.resize(frame, dsize =(480, 480))

    img_mask = fgbg.apply(frame)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('orig', img)
    cv2.imshow('frame', img_mask)

    if cv2.waitKey(30) == 27:
        break

# 카메라가 고정된 상태로 사용해야 한다.