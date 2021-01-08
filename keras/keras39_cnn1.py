# 참고자료: excelsior-cjh.tistory.com/180
# Conv2D(10, (2, 2), input_shape = (5, 5, 1))
# 10: '필터=filter'(output)의 수
# (2, 2): 컨볼루션 '커널'의 (행, 열)
# input_shape = (N, 5, 5, 1): (batch, 행, 열, 채널 수), 채널수: 흑백=1, 컬러=3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

model = Sequential()
# Conv2D는 이미지의 특성을 추출한다. 결과를 보고 한번을 적용할지 두번을 적용할지 그 이상 적용할지 판단한다.
model.add(Conv2D(10, kernel_size = (2,2), strides = 1, padding = 'same', input_shape = (10, 10, 1)))
                                                                                # (N, 10, 10, 1)짜리 이미지를 (2,2)로 자른것
                                                                                # padding은 가장자리 이미지 취급의 취약함을 보완하기위해 사용. 한칸씩 더 늘려준다.
                                                                                # strides는 이미지를 자를때 strides만큼 건너뛰어서 자른다.
model.add(MaxPooling2D(pool_size = (2,3)))                                      # 특성 추출 (2,3) 크기씩 큰 수를 빼내어 특성을 추출한다.
model.add(Conv2D(9, (2, 2), padding = 'valid'))                                 # kernal_size = -> 생략 가능
# model.add(Conv2D(9, (2, 3)))
# model.add(Conv2D(8,2))                                                        # 2 = (2,2)로 인식
model.add(Flatten())                                                            # output shape 1을 맞춰주기 위해
model.add(Dense(1))

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param # => (input_dim * kernel_size + bias(1)) * output 
=================================================================
conv2d (Conv2D)              (None, 9, 9, 10)          50      =>  (1 * 2*2 + 1) * 10
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 9)           369     =>  (10 * 2*2 +1) * 9
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 6, 9)           495
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 5, 8)           296
_________________________________________________________________
flatten (Flatten)            (None, 240)               0
_________________________________________________________________
dense (Dense)                (None, 1)                 241
=================================================================
Total params: 1,451
Trainable params: 1,451
Non-trainable params: 0
'''