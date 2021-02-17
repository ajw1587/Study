import cv2
import numpy as np

def empty(a):
    pass

# 영상 및 이미지에서 손 검출하기
path = '../data/sign_image/my_hand2.mp4' # _image.png

cap = cv2.VideoCapture(path)
if cap.isOpened():
    print('Video Error')

# cv2.namedWindow('TrackBars')
# cv2.resizeWindow('TrackBars', 640, 240)
# # MinMax로 나누는 이뉴는 Min과 Max 사이에 있는 색범위를 추출하기 위해서다.
# cv2.createTrackbar('Hue Min', 'TrackBars', 0, 179, empty)
# cv2.createTrackbar('Hue Max', 'TrackBars', 33, 179, empty)
# cv2.createTrackbar('Sat Min', 'TrackBars', 63, 255, empty)
# cv2.createTrackbar('Sat Max', 'TrackBars', 255, 255, empty)
# cv2.createTrackbar('Val Min', 'TrackBars', 89, 255, empty)
# cv2.createTrackbar('Val Max', 'TrackBars', 255, 255, empty)


while True:     # 원본 -> HSV -> Mor -> contour
    ret, frame = cap.read()
    # img = cv2.imread(path)
    img = cv2.resize(frame, dsize =(480, 480))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h_min = cv2.getTrackbarPos('Hue Min', 'TrackBars')      # 0
    # h_max = cv2.getTrackbarPos('Hue Max', 'TrackBars')      # 33
    # s_min = cv2.getTrackbarPos('Sat Min', 'TrackBars')      # 63
    # s_max = cv2.getTrackbarPos('Sat Max', 'TrackBars')      # 255
    # v_min = cv2.getTrackbarPos('Val Min', 'TrackBars')      # 89
    # v_max = cv2.getTrackbarPos('Val Max', 'TrackBars')      # 255
    h_min = 0
    h_max = 33
    s_min = 63
    s_max = 255
    v_min = 89
    v_max = 255

    # 적용될 색 영역 설정
    # print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    # cv2.imshow('Mask', mask)
    
    # 손 영역내 검은색을 없애기 위한 morphologyEx연산
    kernel = np.ones((5, 5), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    imgMOR = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # contour 설정: 이미지내 흰색 영역을 따라 외곽선을 만들어 준다.
    contours, hierarchy = cv2.findContours(imgMOR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 contour 찾아주기
    max_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_contour = cnt
            max_area = area

    # # 비트 이미지 연산으로 원본 손 추출하기   참고: https://copycoding.tistory.com/156
    imgBGR = cv2.bitwise_and(img, img, mask = imgMOR)
    # # src1: 비교할 이미지
    # # src2: 비교할 이미지
    # # mask: 적용 영역 지정
    # cv2.imshow('imgBGR', imgBGR)

    # contour를 imgBGR 이미지에 적용
    # cv2.drawContours(imgBGR, [max_contour], 0, (255, 0, 0), 3)

    # contour 경계 사각형 imgBGR에 적용시키기
    x, y, w, h = cv2.boundingRect(max_contour)
    rect = cv2.rectangle(imgBGR, (x, y), (x+w, y+h), (0, 255, 0), 1)
    cv2.imshow('Contour-Rectangle', imgBGR)

    # contour 경계 사각형 이미지 잘라주기
    slice_image = rect[y+1:y+h-1, x+1:x+w-1]
    cv2.imshow('Slice Image', slice_image)

    # contour 경계 사각형에 문자 출력

    if cv2.waitKey(30) == 27:
        break