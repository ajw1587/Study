import cv2
import numpy as np

def empty(a):
    pass

path = '../data/lotte/train/0/0.jpg'

cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars', 640, 240)
# MinMax로 나누는 이뉴는 Min과 Max 사이에 있는 색범위를 추출하기 위해서다.
cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
cv2.createTrackbar('Hue Max', 'TrackBars', 0, 179, empty)
cv2.createTrackbar('Sat Min', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('Sat Max', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('Val Min', 'TrackBars', 0, 255, empty)
cv2.createTrackbar('Val Max', 'TrackBars', 0, 255, empty)
img = cv2.imread(path)
img = cv2.resize(img, dsize =(480, 480))
img2 = img.copy()
imgHSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

while True:     # 원본 -> HSV -> Mor -> contour
    h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')      # 0
    h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')      # 33
    s_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')      # 63
    s_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')      # 255
    v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')      # 89
    v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')      # 255

    # 적용될 색 영역 설정
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow('Mask', mask)