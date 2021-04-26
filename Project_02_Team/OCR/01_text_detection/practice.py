from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from matplotlib import pyplot
import numpy as np
import cv2 as cv

model = VGG16(include_top = False, weights = 'imagenet')
model.trainable = False
print(model.summary())

# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
    # get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.output)

# load the image with the required shape
img = load_img('F:/Team Project/OCR/01_Text_detection/Image_data/01.jpg', target_size=(224, 224))

img = img_to_array(img)
img = np.expand_dims(img, axis = 0)
img = preprocess_input(img)

# get feature map for first hidden layer
feature_maps = model.predict(img)

# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()