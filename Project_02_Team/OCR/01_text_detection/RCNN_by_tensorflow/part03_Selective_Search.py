# 출처: https://www.pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/
# 참고: https://linguana.tistory.com/41?category=473528

# import the necessary packages
import argparse
import random
import time
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
ap.add_argument("-m", "--method", type=str, default="fast",
	choices=["fast", "quality"],
	help="selective search method")
args = vars(ap.parse_args())

# args['image'] = 'F:/Team Project/OCR/01_Text_detection/Image_data/ex01.jpg'

# load the input image
image = cv2.imread('F:/Team Project/OCR/01_Text_detection/Image_data/01.jpg')
# initialize OpenCV's selective search implementation and set the
# input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
# check to see if we are using the *fast* but *less accurate* version
# of selective search
if args["method"] == "fast":
	print("[INFO] using *fast* selective search")
	ss.switchToSelectiveSearchFast()
# otherwise we are using the *slower* but *more accurate* version
else:
	print("[INFO] using *quality* selective search")
	ss.switchToSelectiveSearchQuality()


# run selective search on the input image
start = time.time()
rects = ss.process()
end = time.time()
# show how along selective search took to run along with the total
# number of returned region proposals
print("[INFO] selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))

print(type(ss))
# cv2.imshow('image', ss)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# loop over the region proposals in chunks (so we can better
# visualize them)
for i in range(0, len(rects), 100):
	# clone the original image so we can draw on it
	output = image.copy()
	# loop over the current subset of region proposals
	for (x, y, w, h) in rects[i:i + 100]:
		# draw the region proposal bounding box on the image
		color = [random.randint(0, 255) for j in range(0, 3)]
		cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
	# show the output image
	cv2.imshow("Output", output)
	key = cv2.waitKey(0) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break