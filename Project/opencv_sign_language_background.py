import cv2
import numpy as np

def empty(a):
    pass

# 영상 및 이미지에서 손 검출하기
path = '../data/sign_image/my_hand2.mp4' # _image.png

cap = cv2.VideoCapture(path)
if cap.isOpened():
    print('Video Error')

cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 640, 240)
# MinMax로 나누는 이뉴는 Min과 Max 사이에 있는 색범위를 추출하기 위해서다.
cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
cv2.createTrackbar('Hue Max', 'TrackBars', 20, 179, empty)
cv2.createTrackbar('Sat Min', 'TrackBars', 63, 255, empty)
cv2.createTrackbar('Sat Max', 'TrackBars', 166, 255, empty)
cv2.createTrackbar('Val Min', 'TrackBars', 89, 255, empty)
cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)

while True:
    ret, frame = cap.read()
    # img = cv2.imread(path)
    img = cv2.resize(frame, dsize =(480, 480))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')      # 0
    # h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')      # 20
    # s_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')      # 63
    # s_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')      # 166
    # v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')      # 89
    # v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')      # 255
    h_min = 0
    h_max = 20
    s_min = 63
    s_max = 166
    v_min = 89
    v_max = 255

    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)


    # cv2.imshow('Original', img)
    # cv2.imshow('HSV converted', imgHSV)
    cv2.imshow('Mask', mask)
    
    # 비트 이미지 연산으로 원본 확인하기   참고: https://copycoding.tistory.com/156
    imgBGR = cv2.bitwise_or(img, img, mask = mask)
    # src1: 비교할 이미지
    # src2: 비교할 이미지
    # mask: 적용 영역 지정
    cv2.imshow('imgBGR', imgBGR)

    if cv2.waitKey(30) == 27:
        break