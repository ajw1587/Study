import cv2

# OpenCV Image
imagepath = 'C:/data/sign_image/sign_language/asl_dataset/asl_dataset/0/hand1_0_bot_seg_1_cropped.jpeg'
src = cv2.imread(imagepath, cv2.IMREAD_COLOR)

if src is None:
    print('image load failed')

# 이미지의 이진화: 영상의 픽셀 값을 0 or 255로 만드는 연산

cv2.imshow('img', src)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OpenCV Video
# cap = cv2.VideoCapture('../data/video/ggong.mp4')     # 성공하면 True, 실패하면 False

# if not cap.isOpened():
#     print('camera open failed')

# while True():
#     ret, frame = cap.read()                           # return: retval, image
#     if not ret:
#         break
    
#     edge = cv2.Canny(frame, 50, 150)                  # 외곽선 부분만 출력

#     cv2.imshow('frame', frame)
#     cv2.imshow('dege', edge)
#     if cv2.waitKey(1) == 27:                          # waitKey(대기시간ms): 0입력시 무한대기
#                                                       # 동영상마다 frame이 달라 동영상마다 알맞은 값을 지정해줘야한다.
#         break

# cap.release()
# cv2.destroyAllWindows()