# 이미지는
# ata/image/vgg/ 에 4개를 넣으시오
# 개, 고양이, 라이언, 슈트
# 파일명:
# dog1.jpg    cat1.jpg    lion1.jpg    suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('../data/image/vgg/dog1.jpg', target_size = (224, 224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size = (224, 224))
img_lion = load_img('../data/image/vgg/lion1.jpg', target_size = (224, 224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size = (224, 224))
# plt.imshow(img_dog)
# plt.show()

# img -> ndarray
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
# print(type(arr_dog))        # <class 'numpy.ndarray'>
# print(arr_dog.shape)        # (224, 224, 3)

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)

arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
print(arr_input.shape)      # (4, 224, 224, 3)

# 모델 구성
model = VGG16()
results = model.predict(arr_input)
print(results)
print('results.shape: ', results.shape)

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions

decode_results = decode_predictions(results)
print('============================================================')
print('results[0]: ', decode_results[0])
print('============================================================')
print('results[1]: ', decode_results[1])
print('============================================================')
print('results[2]: ', decode_results[2])
print('============================================================')
print('results[3]: ', decode_results[3])
print('============================================================')
# ============================================================
# results[0]:  [('n03255030', 'dumbbell', 0.3473702), ('n04356056', 'sunglasses', 0.07277672), ('n04355933', 'sunglass', 0.05321124), ('n03803284', 'muzzle', 0.025448501), ('n02085620', 'Chihuahua', 0.021639323)]
# ============================================================
# results[1]:  [('n02113186', 'Cardigan', 0.15994819), ('n02123159', 'tiger_cat', 0.12916793), ('n02113023', 'Pembroke', 0.10533797), ('n02123045', 'tabby', 0.08417519), ('n04265275', 'space_heater', 0.056787517)]
# ============================================================
# results[2]:  [('n04548280', 'wall_clock', 0.21537575), ('n03532672', 'hook', 0.14212109), ('n04579432', 'whistle', 0.069210045), ('n02951585', 
# 'can_opener', 0.065023765), ('n03109150', 'corkscrew', 0.059872627)]
# ============================================================
# results[3]:  [('n02102973', 'Irish_water_spaniel', 0.70834917), ('n02105505', 'komondor', 0.16469525), ('n02106382', 'Bouvier_des_Flandres', 0.043103844), ('n02113799', 'standard_poodle', 0.026630424), ('n02093859', 'Kerry_blue_terrier', 0.017082395)]
# ============================================================