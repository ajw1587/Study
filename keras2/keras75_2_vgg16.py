from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

vgg16 = VGG16(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3))
# include_top을 False로 해야 원하는 input 값을 넣을 수 있다.
# print(model.weights)

vgg16.trainable = False
vgg16.summary()
print(len(vgg16.weights))
print(len(vgg16.trainable_weights))

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
model.summary()
print('그냥 가중치의 수: ',len(model.weights))      # 26 -> 32
print('동결하기 전 훈련되는 가중치의 수: ',len(model.trainable_weights))
# 0 -> 6     레이어3개 * (가중치 + 바이어스 = 2) = 6
# 전이학습시 가져온 가중치를 동결시키고 작업