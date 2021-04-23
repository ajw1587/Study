# https://herbwood.tistory.com/11?category=867198
import torch
import torchvision
import torch.nn as nn
import cv2 as cv
from torchvision import transforms
import matplotlib.pyplot as plt

if(torch.cuda.is_available()):
    DEVICE = torch.device('cuda')
    # print(' Device: ', device, '\n', 'Graphic Card: ', torch.cuda.get_device_name(0))
else:
    DEVICE = torch.device('cpu')
    # print(device)


# Feature extraction by pre-trained VGG16
model = torchvision.models.vgg16(pretrained=True).to(DEVICE)
features = list(model.features)

# only collect layers with output feature map size (W, H) < 50
dummy_img = torch.zeros((1, 3, 800, 800)).float() # test image array

req_features = []
output = dummy_img.clone().to(DEVICE)

for feature in features:
    output = feature(output)
#     print(output.size()) => torch.Size([batch_size, channel, width, height])
    if output.size()[2] < 800//16: # 800/16=50
        break
    req_features.append(feature)
    out_channels = output.size()[1]

faster_rcnn_feature_extractor = nn.Sequential(*req_features)

imgTensor = cv.imread('F:/Team Project/OCR/01_Text_detection/Image_data/ga.png')
imgTensor = cv.resize(imgTensor, dsize = (800, 800), interpolation = cv.INTER_CUBIC)

transform = transforms.Compose([transforms.ToTensor()])     # 이미지 변형을 위한 mask
imgTensor = transform(imgTensor).to(DEVICE)
# print('1: ', imgTensor.shape)
imgTensor = imgTensor.unsqueeze(0)                          # 모델에 input하기 위헤 차원 추가
# print('2: ', imgTensor.shape)

output_map = faster_rcnn_feature_extractor(imgTensor)

imgArray = output_map.data.cpu().numpy().squeeze(0)                  # squeeze(0) 차원 없애기
# print(type(imgArray))
# fig = plt.figure(figsize = (12, 4))
# figNo = 1
# for i in range(5):
#     fig.add_subplot(1, 5, figNo)
#     plt.imshow(imgArray[i], cmap = 'gray')
#     figNo += 1
# plt.show()

print('imgArray.shape: ', imgArray.shape)                     # (512, 50, 50)

