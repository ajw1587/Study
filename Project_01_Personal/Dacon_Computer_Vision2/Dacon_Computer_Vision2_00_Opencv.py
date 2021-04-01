import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('../data/csv/Computer_Vision2/train_dirty_mnist_2nd/00001.png')

print(img.shape)
print(type(img))

# 1. Remove Noise

dst = cv2.fastNlMeansDenoisingColored(img,None,18,18,7,21)

# 1. src : 8 비트 3 채널 영상을 입력합니다.
# 2. dst : src와 동일한 크기 및 유형의 출력 이미지.(Output image with the same size and type as src .)
# 3. h : 필터 강도를 조절하는 매개 변수. 
#        값이 클수록 노이즈가 완벽하게 제거되지만 이미지 세부 정보도 제거되고, 
#        값이 작을수록 세부 정보가 보존되지만 일부 노이즈도 보존됩니다.
# 4. hColor: The same as h but for color components. 
#            For most images value equals 10 will be enough to remove colored noise and do not distort colors
# 5. templateWindowSize : 가중치를 계산하는 데 사용되는 템플릿 패치의 크기 (픽셀)입니다. 권장 값 7 픽셀
# 6. searchWindowSize : 주어진 픽셀에 대한 가중 평균을 계산하는 데 사용되는 창의 크기 (픽셀)입니다. 
#                    성능에 선형 적으로 영향을 미침 :
#                    searchWindowsSize 증가-노이즈 제거 시간 증가. 권장 값 21 픽셀

'''
dst = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)

# 1. src: Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image.
# 2. dst: Output image with the same size and type as src .
# 3. templateWindowSize: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels
# 4. searchWindowSize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
# 5. h: Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise

# 출처: https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Non-local_Means_Denoising_Algorithm_Noise_Reduction.php
'''
plt.figure(figsize = (15, 8))
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()