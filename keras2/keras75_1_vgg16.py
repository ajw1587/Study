from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

model = VGG16(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3))
# include_top을 False로 해야 원하는 input 값을 넣을 수 있다.
# print(model.weights)

model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))
'''
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
26
0
'''


# model.trainable = True
# model.summary()
# print(len(model.weights))
# print(len(model.trainable_weights))
'''
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
26
26
'''