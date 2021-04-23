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
print('img0.shape: ', img0.shape)
plt.imshow(img0)
plt.show()

# Object information: a set of bounding boxes [xmin, ymin, xmax, ymax] and their labels
bbox0 = np.array([[161, 152, 242, 232], [626, 314, 695, 394]])
labels = np.array([1, 1])

# display bounding box and labels
img_clone = np.copy(img0)
for i in range(len(bbox0)):
    cv.rectangle(img_clone, (bbox0[i][0], bbox0[i][1]), (bbox0[i][2], bbox0[i][3]), color = (0, 255, 0), thickness = 3)
    cv.putText(img_clone, str(int(labels[i])), (bbox0[i][2], bbox0[i][3]), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness = 3)
plt.imshow(img_clone)
plt.show()


# Resize input image (h = 800, w = 800)
img = cv.resize(img0, dsize = (800, 800), interpolation = cv.INTER_CUBIC)
plt.imshow(img0)
plt.show()

# # change the bounding box coordinates
# Wratio = 800/img0.shape[0]
# Hratio = 800/img0.shape[1]
# ratioLst = [Hratio, Wratio, Hratio, Wratio]
# bbox = []
# for box in bbox0:
#     box = [int(a * b) for a, b in zip(box, ratioLst)]
#     bbox.append(box)
# bbox = np.array(bbox)


# display bounding box and labels
img_clone = np.copy(img)
bbox_clone = bbox0.astype(int)
for i in range(len(bbox0)):
    cv.rectangle(img_clone, (bbox0[i][0], bbox0[i][1]), (bbox0[i][2], bbox0[i][3]), color = (0, 255, 0), thickness = 3)
    cv.putText(img_clone, str(int(labels[i])), (bbox0[i][2], bbox0[i][3]), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), thickness = 3)
plt.imshow(img_clone)
plt.show()

# Use VGG16 to extract features from input images
# Input images(batch_size, H = 800, W = 800, d = 3), Features: (batch_size, H = 50, W = 50, d = 512)
# List all the layers of VGG16
model = torchvision.models.vgg16(pretrained = True).to(device)
fe = list(model.features)
dummy_img = torch.zeros((1, 3, 800, 800)).float()

req_features = []
k = dummy_img.clone().to(device)
for i in fe:                        # i = layer
    k = i(k)
    if k.size()[2] < 800//16:
        break
    req_features.append(i)
    out_channels = k.size()[1]

# Convert this list into a Sequential module
faster_rcnn_fe_extractor = nn.Sequential(*req_features)

# input image
transform = transforms.Compose([transforms.ToTensor()])         # 이미지 변형을 위한 mask
imgTensor = transform(img).to(device)
imgTensor = imgTensor.unsqueeze(0)                              # 모델에 input하기 위헤 차원 추가
out_map = faster_rcnn_fe_extractor(imgTensor)


imgArray = out_map.data.cpu().numpy().squeeze(0)                  # squeeze(0) 차원 없애기
print(type(imgArray))
fig = plt.figure(figsize = (12, 4))
figNo = 1
for i in range(5):
    fig.add_subplot(1, 5, figNo)
    plt.imshow(imgArray[i], cmap = 'gray')
    figNo += 1
plt.show()


# Anchor Box
fe_size = (800//16)     # 50
ctr_x = np.arange(16, (fe_size + 1) * 16, 16)
ctr_y = np.arange(16, (fe_size + 1) * 16, 16)
# [ 16  32  48  64  80  96 112 128 144 160 176 192 208 224 240 256 272 288
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

# display the 2500 anchors
img_clone = np.copy(img)
plt.figure(figsize = (9, 6))
for i in range(ctr.shape[0]):
    cv.circle(img_clone, (int(ctr[i][0]), int(ctr[i][1])), radius = 1, color = (255, 0, 0), thickness = 1)
plt.imshow(img_clone)
plt.show()

# for each of the 2500 anchors, generate 9 anchor boxes
# 2500 x 9 = 22500 anchor boxes
# for i, e in enumerate(ratios):
#     print('np.sqrt(ratios[{}]): '.format(e), np.sqrt(ratios[i]))
#     print('np.sqrt(1./ratios[{}]): '.format(e), np.sqrt(1./ratios[i]))
# np.sqrt(ratios[0.5]):  0.7071067811865476
# np.sqrt(1./ratios[0.5]):  1.4142135623730951
# np.sqrt(ratios[1]):  1.0
# np.sqrt(1./ratios[1]):  1.0
# np.sqrt(ratios[2]):  1.4142135623730951
# np.sqrt(1./ratios[2]):  0.7071067811865476

ratios = [0.5, 1, 2]
scales = [8, 16, 32]
sub_sample = 10# 10
anchor_boxes = np.zeros(((fe_size * fe_size * 9), 4))
index = 0
# print('ctr: ', ctr)
for c in ctr:   # anchors들의 위치
    ctr_x, ctr_y = c
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = sub_sample * scales[j] * np.sqrt(ratios[i])
            w = sub_sample * scales[j] * np.sqrt(1./ratios[i])
            # print('i: ', i, 'j: ', j, 'h, w: ', h, '  ', w)
            anchor_boxes[index, 0] = ctr_x - w / 2.
            anchor_boxes[index, 1] = ctr_y - h / 2.
            anchor_boxes[index, 2] = ctr_x + w / 2.
            anchor_boxes[index, 3] = ctr_y + h / 2.
            index += 1
# print(anchor_boxes.shape)


# # display the 9 anchor boxes of one anchor and the ground truth bbox
# img_clone = np.copy(img)
# for i in range(11025, 11034):       # 0, anchor_boxes.shape[0] 11025, 11034
#     x0 = int(anchor_boxes[i][0])
#     y0 = int(anchor_boxes[i][1])
#     x1 = int(anchor_boxes[i][2])
#     y1 = int(anchor_boxes[i][3])
#     cv.rectangle(img_clone, (x0, y0), (x1, y1), color = (255, 0, 0), thickness = 1)

for i in range(len(bbox)):
    cv.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color = (0, 255, 0), thickness = 3)
plt.imshow(img_clone)
plt.show()


# Ignore cross-boundary anchor boxes
# valid anchor boxes with (x1, y1) > 0 and (x2, y2) <= 800
index_inside = np.where(
    (anchor_boxes[:, 0] >= 0) &
    (anchor_boxes[:, 1] >= 0) &
    (anchor_boxes[:, 2] <= 800) &
    (anchor_boxes[:, 3] <= 800)
)[0]
# print(index_inside.shape)
# print(index_inside)

valid_anchor_boxes = anchor_boxes[index_inside]
# print('valid_anchor_boxes.shape: ', valid_anchor_boxes.shape)

img_clone = np.copy(img)
for i in range(valid_anchor_boxes.shape[0]):       # 0, anchor_boxes.shape[0] 11025, 11034
    x0 = int(valid_anchor_boxes[i][0])
    y0 = int(valid_anchor_boxes[i][1])
    x1 = int(valid_anchor_boxes[i][2])
    y1 = int(valid_anchor_boxes[i][3])
    cv.rectangle(img_clone, (x0, y0), (x1, y1), color = (255, 0, 0), thickness = 1)
plt.imshow(img_clone)
plt.show()

# Calculate iou of the valid anchor boxes
ious = np.empty((len(valid_anchor_boxes), 2), dtype = np.float32)
ious.fill(0)
for num1, i in enumerate(valid_anchor_boxes):
    xa1, ya1, xa2, ya2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        xb1, yb1, xb2, yb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)
        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        if(inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = iter_area / (anchor_area + box_area - iter_area)
        else:
            iou = 0.
        ious[num1, num2] = iou
        # print('\n')
        # print('iou: ', iou)
        # print(type(iou))
        # print('num1: ', num1)
        # print('num2: ', num2)
        # print('ious[num1, num2]: ', ious[num1, num2])
        # print('ious.shape: ', ious.shape)
        # num1: valid_anchor_boxes의 index값
        # num2: bbox의 index값
# print('ious.shape: ', ious.shape)                             # ious.shape:  (13088, 2)
# print('ious[13042, 1]: ', ious[13042, 1])

# What anchor box has max iou with the ground truth bbox
# ious에서 가장 큰 인덱스 위치 얻기
gt_argmax_ious = ious.argmax(axis = 0)
# print('gt_argmax_ious: ', gt_argmax_ious)                     # gt_argmax_ious:  [ 2545 12413]
# print('np.arange(ious.shape[1]): ', np.arange(ious.shape[1])) # np.arange(ious.shape[1]):  [0 1]

# max iou 얻기
# print('ious[2545, 0]: ', ious[2545, 0])                       # ious[2545, 0]:  0.74050635
# print('ious[12413, 1]: ', ious[12413, 1])                     # ious[12413, 1]:  0.7899971
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
# print('gt_max_ious: ', gt_max_ious)                           # gt_max_ious:  [0.74050635 0.7899971 ]


gt_argmax_ious = np.where(ious == gt_max_ious)[0]
# print('np.where(ious == gt_max_ious): ', np.where(ious == gt_max_ious)[0])          
# (array([ 2545,  2552, 12413], dtype=int64), array([0, 0, 1], dtype=int64))
# print('gt_argmax_ious: ', gt_argmax_ious)                     # gt_argmax_ious:  [ 2545  2552 12413]

# print('ious[12413, 1]: ', ious[12413, 1])                     # ious[12413, 1]:  0.7899971
# print('ious[2545, 0]: ', ious[2545, 0])                       # ious[2545, 0]:  0.74050635
# print('ious[2552, 0]: ', ious[2552, 0])                       # ious[2552, 0]:  0.74050635
# print('ious[2552, 1]: ', ious[2552, 1])                       # ious[2552, 1]:  0.0

# What Ground truth bbox is associated with each anchor box
argmax_ious = ious.argmax(axis = 1)
# print('argmax_ious.shape: ', argmax_ious.shape)
# print('argmax_ious: ', argmax_ious)
max_ious = ious[np.arange(len(index_inside)), argmax_ious]
# print('max_ious: ', max_ious)
# print(type(max_ious))
# print(max_ious.shape)


# 13088 valid anchor boxes      1: object, 0: background, -1: ignore
label = np.empty((len(index_inside), ), dtype = np.int32)
label.fill(-1)
# print(label.shape)

# Assign 0 (background) to an anchor if its iou ratio is lower than 0.3 for all ground-truth boxes
pos_iou_threshold = 0.7
neg_iou_threshold = 0.3
label[gt_argmax_ious] = 1
label[max_ious >= pos_iou_threshold] = 1
label[max_ious < neg_iou_threshold] = 0



# mini-batch training 256 valid anchor boxes RPN, 128 positive examples, 128 negetive examples
# mini-batch 사이즈 단위로 label값 변경
n_sample = 256
pos_ratio = 0.5
n_pos = pos_ratio * n_sample            # 128

pos_index = np.where(label == 1)[0]
# print(pos_index)                      # [ 2545  2552 12408 12413]
# print(len(pos_index))                 # 4
if len(pos_index) > n_pos:
    disable_index = np.random.choice(pos_index, size = (len(pos_index) - n_pos), replace = False)
    label[disable_index] = -1

n_neg = n_sample * np.sum(label == 1)   # 4
# print(n_neg)                          # 1024
neg_index = np.where(label == 0)[0]
# print(neg_index)                      # [    0     1     2 ... 13085 13086 13087]
# print(type(neg_index))                # <class 'numpy.ndarray'>
# print(len(neg_index))                 # 12979
if len(neg_index) > n_neg:
    disable_index = np.random.choice(neg_index, size = (len(neg_index) - n_neg), replace = False)
    label[disable_index] = -1


# For each valid anchor box, find the groundtruth object which has max_iou
max_iou_bbox = bbox[argmax_ious]
print(max_iou_bbox.shape)

# valid anchor boxes h, w, cv, cy
height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
ctr_y = valid_anchor_boxes[:, 0] + 0.5 * height
ctr_x = valid_anchor_boxes[:, 1] + 0.5 * width

# valid anchor box max iou bbox h, w, cx, cy
base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

# valid anchor boxes loc = (y-ya/ha), (x-xa/wa), log(h/ha), log(w/wa)
eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
print(anchor_locs.shape)

# torchvision.models.detection.fasterrcnn_resnet50_fpn