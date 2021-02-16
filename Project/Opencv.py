import cv2
import numpy as np

# OpenCV Image
imagepath = 'C:/data/sign_image/sign_language/asl_dataset/asl_dataset/0/hand1_0_bot_seg_1_cropped.jpeg'
src = cv2.imread(imagepath, cv2.IMREAD_COLOR)
'''
if src is None:
    print('image load failed')

# 1. 이미지의 색 변환 #########################################################################
# cv2.cvtColor(): Image Color 변환 색이 필요 없으면 흑백으로 전환하는게 좋다.
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#############################################################################################



# 2. 이미지의 이진화: 영상의 픽셀 값을 0 or 255로 만드는 연산 ####################################
# 밝기에 의해 결과값이 달라질 수 있다. 그래서 thresh값을 자동으로 결정해주는 방법이 있다.
# cv2.THRESH_OTSU: Otsu 알고리즘으로 thresh값 결정
# cv2.THRESH_TRIANGLE: 삼각 알고리즘으로 임계값 결정
th, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)   # thresh 사용 x
print(th)    # Otsu 방법으로 도출해낸 thresh값
# cv2.threshold(src, thresh, maxval, type, dst = None): -> return: retval, dst 
# src: 입력영상 
# thresh: 임계값, thresh를 기준으로 0과 maxval로 나눈다.
# maxval: 최대값, THRESH_BINARY 또는 THRESH_BINARY_INV 방법을 사용할 때 사용.
# type: 임계값에 의한 변환 함ㅅ 지정 또는 자동 임계값 설정 방법 지정 (cv.ThresholdTypes)
# retval: 사용된 임계값
# dst: (출력)임계값 영상 (src와 동일 크기, 동일 타입)
#############################################################################################



# 3. 객체 단위 분석 ##########################################################################
# 3-1. 외곽선 검출 -> 출처: YouTube: 토크ON 71차. 파이썬 OpenCV 입문 IT아카데미
# cv2.findContours(image, mode, method, contours = None, hierarchy = None, offset = None) -> contours, hierarchy(계층정보)
# image: 입력 영상 보통 이진화된 영상을 준다.
# mode: 외곽선 검출 모드 -> RETR_EXTERNAL: 가장 바깥쪽 외곽선 - 계층 정보 x
#                          RETR_LIST: 계층 상관없이 모든 외곽선 검출 - 계층 정보 x
#                          RETR_CCOMP: 2레벨 계츨 구조로 외곽선 검출 - 계층 정보 o
#                          RETR_TREE: 계층적 트리 구조로 모든 외곽선 검출 - 계층 정보 o
# method: CHAIN_APPROX_NONE: 근사화 없음
#         CHAIN_APPROX_SIMPLE: 수직선, 수평선, 대각선에 대해 끝점만 사용하여 압축
#         보통 None을 쓴다.
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contours))            # 외곽선의 개수

for pts in contours:
    if cv2.contourArea(pts) < 1000:
        continue

    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02, True) # approx 점4개의 좌표

    # if len(approx) != 4: # 사각형 형태가 아니면 무시
    #     continue

    cv2.polylines(src, pts, True, (0, 0, 255))
    

# 3-2. 면적 구하기
# cv2.contourArea(contour, oriented = None) -> retval
# contour: 외곽선 좌표
# oriented: True면 외곽선 진행 방향에 따라 부호 있는 면적 반환
# retval: 외곽선으로 구성된 면적

# 3-3. 외곽선 길이 구하기
# cv2.arcLength(curve, closed) -> retval
# curve: 외곽선 좌표
# closed: True이면 폐곡선으로 간주
# retval: 외곽선 길이

# 3-4. 바운딩 박스 (외곽선을 외접하여 둘러싸는 가장 작은 사각형) 구하기
# cv2.boundingRect(array) -> retval
# array: 외곽선 좌표
# retval: 사각형 (x, y, w, h)

# 3-5. 바운딩 서클 (외곽선을 외접하여 둘러싸는 가장 작은 원) 구하기
# cv2.minENclosingCircle(points) -> center, radius
# points: 외곽선 좌표
# center: 바운딩 서클 중심 좌표 (x, y)
# radius: 바운딩 서클 반지름

# 3-6. 외곽선 근사화
# cv2.approxPolyDP(curve, epsilon, closed, approxCUrve = None) -> approxCurve
# curve: 입력 곡선 좌표
# epsilon: 근사화 정밀도 조절, 입력 곡선과 근사화 곡선 간의 최대 거리
# closed: True를 전달하면 폐곡선으로 간주
# approxCurve: 근사화된 곡선 좌표


# 4. 영상의 기하학적 변환 #####################################################################
# 출처: YouTube: 토크ON 71차. 파이썬 OpenCV 입문 IT 아카데미
# 4-1. 투시 변환 행렬 구하기
# cv2.getPerspectiveTransform(src, dst, solveMethod = None) -> retval
# src: 4개의 원본 좌표점   cv2.approxPolyDP 반환값이 들어가는 자리
# dst: 4개의 결과 좌표점
# 반환값: 3x3크리의 투시 변환 행렬

# 4-2. 영상의 투시 변환
# cv2.warpPerspective(src, M, dsize, dst = None, flags = None, borderMode = None, borderValue = None) -> dst
# src: 입력 영상
# M: 3x3 변환 행렬, 실수형, cv2.getPerspectivetransform 반환값이 들어가는 자리
# dsize: 결과 영상의 크기 (size, size)
# flags: 보간법, 기본값은 cv2.INTER_LINEAR
# borderMode: 가장자리 픽셀 확장 방식
# borderValue: cv2.BORDER_CONSTANT 일때 사용할 상수 값, 기본값: 0

#############################################################################################


# OpenCV에서 지원하는 얼굴 검출 기법###########################################################
# 1. Haar Cascade 방법


# 2. DNN 방법: Haar Cascade보다 뛰어난 성능을 보인다.
# 2-1. 네트워크 불러오기
# cv2.dnn.readNet(model, config = none, framework = None) -> retval
# model: 훈련된 가중치를 저장하고 있는 이진 파일 이름
# config: 네트워크 구성을 저장하고 있는 텍스트 파일 이름
# framework: 명시적인 딥러닝 프레임워크 이름
# retval: cv2.dnn_Net 클래스 객체

# 2-2. 네트워크 입력 블롭(blob) 만들기
cv2.dnn.blobFromImage(image, scalefactor = None, size = None, mean = None, swapRB = None,
                      crop = None, ddepth = None) -> retval
# image: 입력 영상
# scalefactor: 입력영상 픽셀 값에 곱할 값
# size: 출력 영상의 크기
# mean: 입력 영상 각 채널에서 뺄 평균값
# swapRB: R과 B 채널을 서로 바꿀 것인지를 결정하는 플래그.
# crop: 크롭(crop) 수행 여부
# ddepth: 출력 블롭의 깊이. CV_32F or CV_8U
# retval: 영상으로부터 구한 블롭 객체
#############################################################################################


cv2.imshow('src', src)
cv2.imshow('src_gray', src_gray)
cv2.imshow('src_bin', src_bin)

cv2.waitKey(0)                  # waitKey(대기시간ms): 0입력시 무한대기
cv2.destroyAllWindows()
'''
#####################################################################################
# OpenCV Video
cap = cv2.VideoCapture('../data/video/my_hand2.mp4')     # 성공하면 True, 실패하면 False
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2)

if not cap.isOpened():
    print('camera open failed')

while True:
    ret, frame = cap.read()                           # return: retval, image
    if not ret:
        break
    
    frame = cv2.resize(frame, (frame_width, frame_height))             # ratio 이상

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, frame_bin = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cv2.imshow('framebin', frame_bin)
    # cv2.imshow('frame_gray', frame_gray)
    # cv2.imshow('frame', frame)

    if cv2.waitKey(30) == 27:                          # waitKey(대기시간ms): 0입력시 무한대기
                                                      # 동영상마다 frame이 달라 동영상마다 알맞은 값을 지정해줘야한다.
        break

cap.release()
cv2.destroyAllWindows()