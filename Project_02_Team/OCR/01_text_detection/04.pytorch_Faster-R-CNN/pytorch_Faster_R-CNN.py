# 출처: https://www.youtube.com/watch?v=4yOcsWg-7g8

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image

if(torch.cuda.is_available()):
    device = torch.device('cuda')
    print(' Device: ', device, '\n', 'Graphic Card: ', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print(device)

# Read Image
img = cv.imread('F:/Team Project/OCR/01_Text_detection/Image_data/ga.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
print('img.shape: ', img.shape)
plt.imshow(img)
plt.show()

# Object information: a set of bounding boxes [xmin, ymin, xmax, ymax] and their labels
bbox = np.array([[161, 152, 242, 232], [626, 314, 695, 394]])
labels = np.array([1, 1])

# display bounding box and labels
img_clone = np.copy(img)
for i in range(len(bbox)):
    cv.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color = (0, 255, 0), thickness = 3)
    cv.putText(img_clone, str(int(labels[i])), (bbox[i][2], bbox[i][3]), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness = 3)
plt.imshow(img_clone)
plt.show()