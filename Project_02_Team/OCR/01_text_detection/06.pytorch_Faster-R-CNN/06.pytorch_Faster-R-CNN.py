# https://herbwood.tistory.com/11?category=867198
# nuggy875.tistory.com/33
import torch
import torchvision
import torch.nn as nn
import cv2 as cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

if(torch.cuda.is_available()):
    DEVICE = torch.device('cuda')
    # print(' Device: ', device, '\n', 'Graphic Card: ', torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device('cpu')
    # print(device)

# image
img0 = cv2.imread('F:/Team Project/OCR/01_Text_detection/data/faster_rcnn/test/zebras.jpg')
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

bbox0 = np.array([[223, 782, 623, 1074], [597, 695, 1038, 1050], 
                  [1088, 699, 1452, 1057], [1544, 771, 1914, 1063]])

# resize img, bbox
img = cv2.resize(img0, dsize = (800, 800), interpolation = cv2.INTER_CUBIC)

Wratio = 800 / img0.shape[1]
Hratio = 800 / img0.shape[0]

ratioList = [Wratio, Hratio, Wratio, Hratio]
bbox = []
for box in bbox0:
    box = [int(a * b) for a, b in zip(box, ratioList)]
    bbox.append(box)

# # Show img, bbox
# img_clone = np.copy(img)
# for i in range(len(bbox)):
#     cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color=(0, 255, 0), thickness=5)
# plt.imshow(img_clone)
# plt.show()


# Model
model = torchvision.models.vgg16(pretrained = True).to(DEVICE)
features = list(model.features)

# only collect layers with output feature map size (W, H) < 50
dummy_img = torch.zeros((1, 3, 800, 800)).float()

req_features = []
output = dummy_img.clone().to(DEVICE)

for feature in features:
    output = feature(output)
    print(output.size())

    if output.size()[2] < 800 / 16:
        break

    req_features.append(feature)
    out_channels = output.size()[1]

faster_rcnn_feature_extractor = nn.Sequential(*req_features)

# cvt to tensor
transform = transforms.Compose([transforms.ToTensor()])
imgTensor = transform(img).to(DEVICE)
imgTensor = imgTensor.unsqueeze(0)
output_map = faster_rcnn_feature_extractor(imgTensor)

# # visualize the first 5 channels of the 50*50*512 feature maps
# imgArray = output_map.data.cpu().numpy().squeeze(0)
# fig = plt.figure(figsize=(12, 4))
# figNo = 1
# for i in range(5):
#     fig.add_subplot(1, 5, figNo)
#     plt.imshow(imgArray[i])
#     figNo += 1
# plt.show()


# Generate Anchors
# sub-sampling rate = 1/16
# image size : 800x800
# sub-sampled feature map size : 800 x 1/16 = 50
# 50 x 50 = 2500 anchors and each anchor generate 9 anchor boxes
# total anchor boxes = 50 x 50 x 9 = 22500
# x,y intervals to generate anchor box center

feature_size = 800 // 16
ctr_x = np.arange(16, (feature_size + 1) * 16, 16)
ctr_y = np.arange(16, (feature_size + 1) * 16, 16)
print(len(ctr_x))
print(ctr_x)

# coordinates of the 255 center points to generate anchor boxes
index = 0
ctr = np.zeros((2500, 2))

for i in range(len(ctr_x)):
    for j in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[i] - 8
        ctr[index, 0] = ctr_y[j] - 8
        index += 1

# ctr => [[center x, center y], ...]
print('ctr: ', ctr.shape)
print('ctr: ', ctr[:10, :])

# display the 2500 anchors within image
img_clone2 = np.copy(img)
ctr_int = ctr.astype("int32")

plt.figure(figsize=(7, 7))
for i in range(ctr.shape[0]):
    cv2.circle(img_clone2, (ctr_int[i][0], ctr_int[i][1]),
              radius=1, color=(255, 0, 0), thickness=3)
plt.imshow(img_clone2)
plt.show()


# for each of the 2500 anchors, generate 9 anchor boxes
# 2500 x 9 = 22500 anchor boxes
ratios = [0.5, 1, 2]
scales = [8, 16, 32]
sub_sample = 16

anchor_boxes = np.zeros(((feature_size * feature_size * 9), 4))     # 50 * 50개의 anghorbox
index = 0

for c in ctr:                        # per anchors
    ctr_y, ctr_x = c
    for i in range(len(ratios)):     # per ratios
        for j in range(len(scales)): # per scales
            
            # anchor box height, width
            h = sub_sample * scales[j] * np.sqrt(ratios[i])
            w = sub_sample * scales[j] * np.sqrt(1./ ratios[i])
            
            # anchor box [x1, y1, x2, y2]
            anchor_boxes[index, 1] = ctr_y - h / 2.
            anchor_boxes[index, 0] = ctr_x - w / 2.
            anchor_boxes[index, 3] = ctr_y + h / 2.
            anchor_boxes[index, 2] = ctr_x + w / 2.
            index += 1
print('anfhor: ', anchor_boxes.shape)
print('anfhor: ', anchor_boxes[:10, :])


# display the anchor boxes of one anchor and the ground truth boxes
img_clone = np.copy(img)

# draw random anchor boxes
for i in range(11025, 11034):
    x1 = int(anchor_boxes[i][0])
    y1 = int(anchor_boxes[i][1])
    x2 = int(anchor_boxes[i][2])
    y2 = int(anchor_boxes[i][3])
    
    cv2.rectangle(img_clone, (x1, y1), (x2, y2), color=(255, 0, 0),
                 thickness=3)

# draw ground truth boxes
for i in range(len(bbox)):
    cv2.rectangle(img_clone, (bbox[i][0], bbox[i][1]), 
                             (bbox[i][2], bbox[i][3]),
                 color=(0, 255, 0), thickness=3)

plt.imshow(img_clone)
plt.show()

# ignore the cross-boundary anchor boxes
# valid anchor boxes with (x1, y1) > 0 and (x2, y2) <= 800
index_inside = np.where(
        (anchor_boxes[:, 0] >= 0) &
        (anchor_boxes[:, 1] >= 0) &
        (anchor_boxes[:, 2] <= 800) &
        (anchor_boxes[:, 3] <= 800))[0]

print(index_inside.shape)

# only 8940 anchor boxes are inside the boundary out of 22500
valid_anchor_boxes = anchor_boxes[index_inside]
print(valid_anchor_boxes.shape)


# calculate Iou of the valid anchor boxes
# since we have 8940 anchor boxes and 4 ground truth objects,
# we should get an array with (8940, 4) as the output
# [IoU with gt box1, IoU with gt box2, IoU with gt box3,IoU with gt box4]
ious = np.empty((len(valid_anchor_boxes),4), dtype=np.float32)
ious.fill(0)

# anchor boxes
for i, anchor_box in enumerate(valid_anchor_boxes):
    xa1, ya1, xa2, ya2 = anchor_box
    anchor_area = (xa2 - xa1) * (ya2 - ya1)
    
    # ground truth boxes
    for j, gt_box in enumerate(bbox):
        xb1, yb1, xb2, yb2 = gt_box
        box_area = (xb2 - xb1) * (yb2 - yb1)
        
        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])
        
        if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            iou = inter_area / (anchor_area + box_area - inter_area)
        else:
            iou = 0
        # print('i: ', i)
        # print('j: ', j)
        ious[i, j] = iou
        # print('ious[i, j]: ', ious[i, j])
        # print('ious.shape: ', ious.shape)
print('ious: ', ious.shape)

# Sample positive/negative anchor boxes
# what anchor box has max ou with the ground truth box
gt_argmax_ious = ious.argmax(axis=0)
print('gt_argmax_ious: ', gt_argmax_ious)

gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
print('gt_max_ious: ', gt_max_ious)

gt_argmax_ious = np.where(ious == gt_max_ious)[0]
print('gt_argmax_ious: ', gt_argmax_ious)


# what ground truth bbox is associated with each anchor box
argmax_ious = ious.argmax(axis=1)
print('argmax_ious.shape: ', argmax_ious.shape)
print('argmax_ious: ', argmax_ious)

max_ious = ious[np.arange(len(index_inside)), argmax_ious]
print('max_ious: ', max_ious)