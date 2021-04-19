from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from part01_sliding_window_image_pyramid import sliding_window
from part01_sliding_window_image_pyramid import image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(200, 150)",
	help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9,
	help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,
	help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())
# --image: 객체 탐지를 할 이미지의 경로
# --size: 튜플, sliding window의 크기. 쌍따옴표로 둘러싸야 함 (e.g. "(200, 150)")
# --min-cof: bounding boxes를 걸러줄 최소 신뢰도 값
# --visualize: 디버깅을 위해 추가적으로 시각화를 해줄지 여부

# initialize variables used for the object detection procedure
WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224, 224)
# WIDTH: 각 이미지가 같은 사이즈가 아니기 때문에 시작할 때의 사이즈는 일정하도록 고정시킴
# PYR_SCALE: image_pyramid 함수의 scale factor. 각 층마다 어느 비율로 크기가 조정되는지 나타냄. 값이 작을수록 더 많은 층이 생기고, 값이 클수록 적은 층이 생김. 층이 더 적을 수록 전반적인 객체 탐지 과정은 빠르게 작동하지만, 정확도는 떨어질 수도 있음
# WIN_STEP: sliding window 가 x, y 방향으로 skip하는 픽셀의 값. 이 값이 작을수록 더 많은 windows를 검토해야 하고 더 느려짐. 실질적으로 4 혹은 8 정도의 값이 적당함. (입력 이미지의 크기나 minSize의 값에 따라서 좀 다르긴 함)
# ROI_SIZE: 탐지하고 싶은 객체의 크기 비율을 조절함. 만약에 이 비율에서 문제가 있으면 객체를 탐지하는 건 거의 불가능임. 추가적으로 이 값은 minSize 값과 연관이 되어있음. (minSize는 반복문에서 탈출하는 것을 결정한다는 걸 기억할 것) ROI_SIZE는 cmd에 주어질 argument 중 --size와 직접적으로 연관됨.
# INPUT_SIZE: CNN 분류기의 차원 크기. 이 값은 당신이 사용할 CNN에 따라서 크게 달라질 수 있음. (우리의 경우 ResNet50를 사용함)

# load our network weights from disk
print("{INFO} loading network...")
model = ResNet50(weights = 'imagenet', include_top = True)
# load the input image from disk, resize it such that it has the
# has the supplied width, and then grab its dimensions
orig = cv2.imread(args["image"])
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]


# initialize the image pyramid
pyramid = image_pyramid(orig, scale = PYR_SCALE, minSize = ROI_SIZE)

# initialize two lists, one to hold the ROIs generated from the image
# pyramid and sliding window, and another list used to store the
# (x, y)-coordinates of where the ROI was in the original image
rois = []
locs = []
# rois: pyramid와 sliding window의 출력으로 만들어진 ROI(이미지)를 담는다
# locs: ROI가 원본 이미지에 있었던 (x, y) 좌표를 담는다

# time how long it takes to loop over the image pyramid layers and
# sliding window locations
start = time.time()

# loop over the image pyramid
for image in pyramid:
	# determine the scale factor between the *original* image
	# dimensions and the *current* layer of the pyramid
	scale = W / float(image.shape[1])
	# for each layer of the image pyramid, loop over the sliding
	# window locations
	for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
		# scale the (x, y)-coordinates of the ROI with respect to the
		# *original* image dimensions
		x = int(x * scale)
		y = int(y * scale)
		w = int(ROI_SIZE[0] * scale)
		h = int(ROI_SIZE[1] * scale)
		# take the ROI and preprocess it so we can later classify
		# the region using Keras/TensorFlow
		roi = cv2.resize(roiOrig, INPUT_SIZE)
		roi = img_to_array(roi)
		roi = preprocess_input(roi)
		# update our list of ROIs and associated coordinates
		rois.append(roi)
		locs.append((x, y, x + w, y + h))

        # check to see if we are visualizing each of the sliding
		# windows in the image pyramid
		if args["visualize"] > 0:
			# clone the original image and then draw a bounding box
			# surrounding the current region
			clone = orig.copy()
			cv2.rectangle(clone, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
                
			# show the visualization and current ROI
			cv2.imshow("Visualization", clone)
			cv2.imshow("ROI", roiOrig)
			cv2.waitKey(0)

# show how long it took to loop over the image pyramid layers and
# sliding window locations
end = time.time()
print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
	end - start))
# convert the ROIs(Window Sliding으로 자른 이미지 리스트) to a NumPy array
rois = np.array(rois, dtype="float32")
# classify each of the proposal ROIs using ResNet and then show how
# long the classifications took
print("[INFO] classifying ROIs...")
start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] classifying ROIs took {:.5f} seconds".format(
	end - start))
# decode the predictions and initialize a dictionary which maps class
# labels (keys) to any ROIs associated with that label (values)
preds = imagenet_utils.decode_predictions(preds, top=1)
# decode_predictions: https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/preprocess_input?authuser=19&hl=ko
# decode_predictions: 라벨의 확률 계산기
labels = {}

# label 정리해주기
# loop over the predictions
for (i, p) in enumerate(preds):
	# grab the prediction information for the current ROI
	(imagenetID, label, prob) = p[0]
	# filter out weak detections by ensuring the predicted probability
	# is greater than the minimum probability
	if prob >= args["min_conf"]:    # 신뢰도 검사
		# grab the bounding box associated with the prediction and
		# convert the coordinates
		box = locs[i]
		# grab the list of predictions for the label and add the
		# bounding box and probability to the list
		L = labels.get(label, [])
		L.append((box, prob))
		labels[label] = L


# loop over the labels for each of detected objects in the image
for label in labels.keys():
	# clone the original image so that we can draw on it
	print("[INFO] showing results for '{}'".format(label))
	clone = orig.copy()
	# loop over all bounding boxes for the current label
	for (box, prob) in labels[label]:
		# draw the bounding box on the image
		(startX, startY, endX, endY) = box
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
	# show the results *before* applying non-maxima suppression, then
	# clone the image again so we can display the results *after*
	# applying non-maxima suppression
	cv2.imshow("Before", clone)
	clone = orig.copy()


    # extract the bounding boxes and associated prediction
	# probabilities, then apply non-maxima suppression
	boxes = np.array([p[0] for p in labels[label]])
	proba = np.array([p[1] for p in labels[label]])
	boxes = non_max_suppression(boxes, proba)
	# loop over all bounding boxes that were kept after applying
	# non-maxima suppression
	for (startX, startY, endX, endY) in boxes:
		# draw the bounding box and label on the image
		cv2.rectangle(clone, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.putText(clone, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	# show the output after apply non-maxima suppression
	cv2.imshow("After", clone)
	cv2.waitKey(0)