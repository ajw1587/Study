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
    # print(' Device: ', device, '\n', 'Graphic Card: ', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    # print(device)

# Read Image
img0 = cv.imread('F:/Team Project/OCR/01_Text_detection/Image_data/ga.png')
img0 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
# print('img0.shape: ', img0.shape)
# plt.imshow(img0)
# plt.show()

# Object information: a set of bounding boxes [xmin, ymin, xmax, ymax] and their labels
bbox0 = np.array([[161, 152, 242, 232], [626, 314, 695, 394]])
labels = np.array([1, 1])

# display bounding box and labels
img_clone = np.copy(img0)
for i in range(len(bbox0)):
    cv.rectangle(img_clone, (bbox0[i][0], bbox0[i][1]), (bbox0[i][2], bbox0[i][3]), color = (0, 255, 0), thickness = 3)
    cv.putText(img_clone, str(int(labels[i])), (bbox0[i][2], bbox0[i][3]), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness = 3)
# plt.imshow(img_clone)
# plt.show()


# Resize input image (h = 800, w = 800)
img = cv.resize(img0, dsize = (800, 800), interpolation = cv.INTER_CUBIC)
# plt.imshow(img0)
# plt.show()

# change the bounding box coordinates
Wratio = 800/img0.shape[0]
Hratio = 800/img0.shape[1]
# print(img0.shape[0])
# print(img0.shape[1])
# print(Wratio)
# print(Hratio)
ratioLst = [Hratio, Wratio, Hratio, Wratio]
# print(ratioLst)
bbox = []
for box in bbox0:
    box = [int(a * b) for a, b in zip(box, ratioLst)]
    bbox.append(box)
bbox = np.array(bbox)
# print(bbox)


# display bounding box and labels
img_clone = np.copy(img)
bbox_clone = bbox.astype(int)
for i in range(len(bbox)):
    cv.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color = (0, 255, 0), thickness = 3)
    cv.putText(img_clone, str(int(labels[i])), (bbox[i][2], bbox[i][3]), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness = 3)
# plt.imshow(img_clone)
# plt.show()

# Use VGG16 to extract features from input images
# Input images(batch_size, H = 800, W = 800, d = 3), Features: (batch_size, H = 50, W = 50, d = 512)
# List all the layers of VGG16
model = torchvision.models.vgg16(pretrained = True).to(device)
fe = list(model.features)
# print('fe: ', fe)
# print(len(fe))

# collect layers with output feature map size (W, H) < 50
dummy_img = torch.zeros((1, 3, 800, 800)).float()
# print(dummy_img.shape)

req_features = []
k = dummy_img.clone().to(device)
for i in fe:                        # i = layer
    k = i(k)
    print(type(i))
    print('###################### ', i, ' ######################')
    print(type(k))
    print(k.shape)
    if k.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = k.size()[1]
print('len(req_features): ', len(req_features))
print('out_channels: ', out_channels)

# Convert this list into a Sequential module
faster_rcnn_fe_extractor = nn.Sequential(*req_features)

# input image
transform = transforms.Compose([transforms.ToTensor()])     # 이미지 변형을 위한 mask
imgTensor = transform(img).to(device)
# print('1: ', imgTensor.shape)
imgTensor = imgTensor.unsqueeze(0)                          # 모델에 input하기 위헤 차원 추가
# print('2: ', imgTensor.shape)
out_map = faster_rcnn_fe_extractor(imgTensor)
# print('out_map.size(): ', out_map.size())

# visualize the first 5 channels of the 50 * 50 * 512 feature maps
# print(out_map.data.shape)                                   # torch.Size([1, 512, 50, 50])
# print(out_map.data.cpu().shape)                             # torch.Size([1, 512, 50, 50])
# print(out_map.data.numpy().shape)                           # (1, 512, 50, 50)
# print(out_map.data.numpy().squeeze(0).shape)                # (512, 50, 50)


# imgArray = out_map.data.cpu().numpy().squeeze(0)                  # squeeze(0) 차원 없애기
# print(type(imgArray))
# fig = plt.figure(figsize = (12, 4))
# figNo = 1
# for i in range(5):
#     fig.add_subplot(1, 5, figNo)
#     plt.imshow(imgArray[i], cmap = 'gray')
#     figNo += 1
# plt.show()


# Anchor Box
# Generate 22,500 anchor boxes on each input image
# 50 x 50 = 2500 anchors, each anchor generate 9 anchor boxes, Total = 50 x 50 x 9 = 22,500
# x, y intervals to generate anchor box center
fe_size = (800//16)     # 50
print(fe_size)
ctr_x = np.arange(16, (fe_size + 1) * 16, 16)
ctr_y = np.arange(16, (fe_size + 1) * 16, 16)
# print(len(ctr_x), ctr_x)
# 50 [ 16  32  48  64  80  96 112 128 144 160 176 192 208 224 240 256 272 288
#  304 320 336 352 368 384 400 416 432 448 464 480 496 512 528 544 560 576
#  592 608 624 640 656 672 688 704 720 736 752 768 784 800]


# coordinates of the 2500 center points to generate anchor boxes
index = 0
ctr = np.zeros((2500, 2))
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index, 0] = ctr_x[x] - 8
        ctr[index, 1] = ctr_y[y] - 8
        index += 1
# print(ctr)
# print(ctr.shape)

# # display the 2500 anchors
# img_clone = np.copy(img)
# plt.figure(figsize = (9, 6))
# for i in range(ctr.shape[0]):
#     cv.circle(img_clone, (int(ctr[i][0]), int(ctr[i][1])), radius = 1, color = (255, 0, 0), thickness = 1)
# plt.imshow(img_clone)
# plt.show()

# for each of the 2500 anchors, generate 9 anchor boxes
# 2500 x 9 = 22500 anchor boxes
ratios = [0.5, 1, 2]# [0.5, 1, 2]
scales = [8, 16, 32]# [4, 8, 16]
sub_sample = 10
anchor_boxes = np.zeros(((fe_size * fe_size * 9), 4))
index = 0
print('ctr: ', ctr)
for c in ctr:   # anchors들의 위치
    ctr_x, ctr_y = c
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = sub_sample * scales[j] * np.sqrt(ratios[i])
            w = sub_sample * scales[j] * np.sqrt(1./ratios[i])
            print('i: ', i, 'j: ', j, 'h, w: ', h, '  ', w)
            anchor_boxes[index, 0] = ctr_x - w / 2.
            anchor_boxes[index, 1] = ctr_y - h / 2.
            anchor_boxes[index, 2] = ctr_x + w / 2.
            anchor_boxes[index, 3] = ctr_y + h / 2.
            index += 1
# print(anchor_boxes.shape)


# display the 9 anchor boxes of one anchor and the ground trugh bbox
img_clone = np.copy(img)
for i in range(11025, 11034):       # 0, anchor_boxes.shape[0]
    x0 = int(anchor_boxes[i][0])
    y0 = int(anchor_boxes[i][1])
    x1 = int(anchor_boxes[i][2])
    y1 = int(anchor_boxes[i][3])
    cv.rectangle(img_clone, (x0, y0), (x1, y1), color = (255, 0, 0), thickness = 1)

for i in range(len(bbox)):
    cv.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color = (0, 255, 0), thickness = 3)

plt.imshow(img_clone)
plt.show()