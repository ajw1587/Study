# 출처: https://www.dacon.io/competitions/open/235560/codeshare/613?page=1&dtype=recent

import random
import pprint
import sys
import time
import numpy as np
import pickle
import math
import cv2
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.layers import Layer
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

from utility import *

########################################################################################################
# Write train.csv to annotation.txt
# train.csv 파일 안의 내용을 txt 파일로 변경합니다. 이 txt파일은 나중에 데이터를 파싱하는 과정에서 사용됩니다.

# txt 파일은 filename, x1, x2, y1, y2, classname을 ,로 구분하여 저장합니다. 
# 여기서 x1, x2, y1, y2는 바운딩 박스의 좌표입니다.
base_path = os.getcwd()     # 현재 디렉토리의 이름을 가져온다.
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))

# For training
f= open(base_path + "/annotation_train.txt","w+")
for idx, row in train_df.iterrows():
    sys.stdout.write('Parse train_imgs ' + str(idx) + '; Number of boxes: ' + str(len(train_df)) + '\r')
    sys.stdout.flush()
    x1 = int(row['bbox_x1'])
    x2 = int(row['bbox_x2'])
    y1 = int(row['bbox_y1'])
    y2 = int(row['bbox_y2'])
    className = row['class']
    fileName = os.path.join(base_path, 'train', row['img_file'])

    f.write(str(fileName) + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(className) + '\n')
f.close()
########################################################################################################



########################################################################################################
# Build pretrained model
# 사전 훈련된 모델을 적재합니다. 이 노트북은 ResNet50을 사전 훈련된 모델로 사용하며, 
# activation_39층을 feature map으로 사용합니다. 모델을 정의한 후, 가중치들을 저장합니다. 
# 이는 추후 RPN과 Classification layer에서 사용됩니다.

from tensorflow.keras.applications.resnet50 import ResNet50

resnet50 = ResNet50(weights='imagenet', include_top=False)
outputs = resnet50.get_layer('activation_39').output

model = Model(resnet50.input, outputs)

#resnet50.save_weights(os.path.join(base_path, 'model/', 'resnet50.h5')) # save pre-trained model to load weights later
########################################################################################################



########################################################################################################
# RPN layer
# RPN(Region Proposal Network) 층을 정의합니다.
# 먼저 사전 훈련된 모델의 feature map을 3x3x1024 합성곱층에 통과시킵니다. 
# 그 후 이 값을 두개의 1x1 합성곱층에 통과시켜 하나는 물체가 있는지 없는지를 판별하는 분류기, 
# 다른 하나는 바운딩 박스의 좌표를 예측하는 회귀모델로 사용합니다.
# 이때 Faster R-CNN은 앵커라는 개념을 도입하는데 앵커는 사전에 크기와 가로 세로 비율이 정의된 박스입니다. 
# 논문에서는 크기가 [128, 256, 512], 가로 세로 비율이 [1:1, 1:2, 2:1] 인 총 9개의 앵커를 사용합니다. 
# 그렇기 때문에 1x1 합성곱층을 통과한 후 정의되는 분류기 x_class는 9개 클래스를 가지며, 
# 시그모이드 활성화 함수를 사용하여 0 또는 1을 출력합니다.
# 예를 들어 x_class 결과값이 [1, 0, 0, 1, 0, 0, 0, 0, 0] 이라면, 
# 이는 첫번째와 네번째 앵커에 물체를 포함하고 있음을 나타냅니다.
# 이와 유사하게 바운딩 박스의 좌표를 조절하는 회귀모델 x_regr은 
# 앵커의 수(9개) x 바운딩 박스의 좌표(x1, x2, y1, y2) 총 36개의 클래스를 예측합니다. 
# 활성화 함수는 렐루를 사용합니다.

def rpn_layer(base_layers, num_anchors):
    """Create a rpn layer
        Step1: Pass through the feature map from base layer to a 3x3 1024 channels convolutional layer
                Keep the padding 'same' to preserve the feature map's size
        Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
    Args:
        base_layers: resnet50 in here
        num_anchors: 9 in here

    Returns:
        [x_class, x_regr, base_layers]
        x_class: classification for whether it's an object
        x_regr: bboxes regression
        base_layers: resnet50 in here
    """
    x = Conv2D(1024, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='relu', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]
########################################################################################################



########################################################################################################
# Define ROI Pooling Layer
# RoI Pooling 층은 Classifier layer의 첫번째층이며, 
# RPN층에서 선정된 지역들(Regions)의 크기를 조절하는 층입니다. 
# 예를 들어 (224, 224, 3) 크기의 이미지를 사전 훈련된 모델에 통과시켜 
# activation_39층에서 얻어지는 feature map의 크기는 (14, 14, 1024)입니다. 
# 이 feature map이 RPN을 통과한 후 예측된 지역의 바운딩 박스의 크기는 랜덤하게 
# 10x13, 11x12, 9x13 등으로 각기 다를수 있습니다. 
# RoI Pooling 층은 이러한 바운딩 박스의 크기를 7x7로 고정시켜주는 역할을 합니다.

class RoiPoolingConv(Layer):
    '''ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    '''
    def __init__(self, pool_size, num_rois, **kwargs):

        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]   

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert(len(x) == 2)

        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        input_shape = K.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            # Resized roi of the image to pooling size (7x7)
            rs = tf.image.resize(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)
                

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))

        return final_output
    
    
    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'num_rois': self.num_rois}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
########################################################################################################



########################################################################################################
# Classifier layer
# Classifer layer(Fast R-CNN)을 정의합니다. 
# 이 층은 ResNet50의 feature map(base_layers), 
# RPN에서 제안된 Rois(input_rois), 
# 한번에 이미지에 투영할 Rois의 수(num_rois)를 입력으로 받습니다. 
# base_layers의 shape은 (224, 224, 3)의 이미지를 입력으로하였을 경우 (None, 14, 14, 1024)이며, 
# input_rois는 (None, num_rois, 4)의 shape를 가지고있습니다. 
# 논문에서는 Feature map을 RoIPooling 층, 두 개의 4096 Fully-connected층에 통과시킨후 이미지 분류와 바운딩 박스 회귀를 진행합니다.
'''
     이 노트북에서 사용한 feature map activation_39층은 ResNet50에서 4번째 합성곱층의 출력입니다. 
     실제 ResNet50은 5개의 합성곱층으로 구성되어있고 Mask R-CNN 논문에서는 Faster R-CNN / Resnet50 모델을 사용할때 activation_39층 → RoIPooling 층 → ResNet_conv5층 → GlobalAveragePooling2D층 → Dense(2048)층으로 모델을 만듭니다.
     이 노트북에서는 ResNet_conv5층을 생략하고 activation_39층 → RoIPooling 층 → GlobalAveragePooling2D층 → Dense(2048)층으로 모델을 만듭니다. 추후 ResNet_conv5를 포함하여 모델을 다시 훈련해보겠습니다.
'''
# Classifer layer에서 한가지 더 주목해야하는 것은 TimeDistributed라는 래퍼(wrapper)로 합성곱층을 감싸고있는것입니다. 
# TimeDistributed은 RNN 모델에서 주로 사용되는 래퍼인데 3차원 이상의 입력값을 마스킹할 수 있습니다. 
# Classifer layer는 기존의 입력에 input_rois를 추가해서 입력을 받기 때문에 하나의 차원이 더 추가되어 TimeDistributed 래퍼를 사용해 주어야합니다.

def classifier_layer(base_layers, input_rois, num_rois, nb_classes):
    """Create a classifier layer
    
    Args:
        base_layers: ResNet50
        input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        num_rois: number of rois to be processed in one time (4 in here)

    Returns:
        list(out_class, out_regr)
        out_class: classifier layer output
        out_regr: regression layer output
    """

    pooling_regions = 7

    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])


    out = TimeDistributed(GlobalAveragePooling2D())(out_roi_pool)
    out = TimeDistributed(Dense(2048, activation='relu', kernel_initializer='he_normal', name='fc1'))(out)
    #out = TimeDistributed(Dropout(0.25))(out)
    #out = TimeDistributed(Dense(4096, activation='relu', kernel_initializer='he_normal', name='fc2'))(out)
    #out = TimeDistributed(Dropout(0.5))(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='lecun_normal'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='relu'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]
########################################################################################################



########################################################################################################
# Define loss functions
# Faster R-CNN은 총 4개의 손실 함수를 가지고 있습니다. RPN에서 분류기와 회귀모델, Classifier layer에서 물체를 분류하는 분류기와 회귀모델입니다. 이 노트북에서는 논문과 동일하게 다음의 손실 함수를 사용합니다.
'''
    RPN 분류기 : Binary_crossentropy
    RPN 회귀 모델 : Smooth L1
    Classifier 분류기 : CategoricalCrossentropy
    Classifier 회귀 모델 : Smooth L1
'''
lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_regr_fixed_num(y_true, y_pred):

        # x is the difference between true value and predicted vaue
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred

        # absolute value of x
        x_abs = K.abs(x)

        # If x_abs &lt;= 1.0, x_bool = 1
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])


def rpn_loss_regr(num_anchors):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs &lt; 1)
                           x_abx - 0.5 (otherwise)
    """
    return rpn_loss_regr_fixed_num
  
def rpn_loss_cls_fixed_num(y_true, y_pred):

            return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])


def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor =&gt; isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    

    return rpn_loss_cls_fixed_num

def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])

def class_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs &lt; 1)
                           x_abx - 0.5 (otherwise)
    """ 
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    cal = CategoricalCrossentropy()
    loss = cal(y_true[0, :, :], y_pred[0, :, :])
    return lambda_cls_class * K.mean(loss)
########################################################################################################



# Start training
########################################################################################################
base_path = '../input/pretrained-model/'
annotation_path = '../input/annotation/'

train_path =  os.path.join(annotation_path, 'annotation_train.txt') # Training data (annotation file)

num_rois = 4 # Number of RoIs to process at once.

# Augmentation flag
horizontal_flips = False # Augment with horizontal flips in training. 
vertical_flips = False   # Augment with vertical flips in training. 
rot_90 = False           # Augment with 90 degree rotations in training. 

output_weight_path = os.path.join(base_path, 'model_frcnn_res50_rpn.h5')

rpn_weight_path = os.path.join(base_path, 'model_frcnn_res50_rpn.h5')

cls_weight_path = os.path.join(base_path, 'model_frcnn_res50_cls.h5')

record_path = os.path.join(base_path, 'record.csv') # Record data (used to save the losses, classification accuracy and mean average precision)

base_weight_path = os.path.join(base_path, 'resnet50.h5')

#config_output_filename = os.path.join(base_path, 'model_res_config.pickle')
########################################################################################################

########################################################################################################
# Create the config
C = Config()

C.use_horizontal_flips = horizontal_flips
C.use_vertical_flips = vertical_flips
C.rot_90 = rot_90

C.record_path = record_path
C.model_path = output_weight_path
C.model_rpn_path = rpn_weight_path
C.model_cls_path = cls_weight_path
C.num_rois = num_rois

C.base_net_weights = base_weight_path
########################################################################################################

########################################################################################################
#--------------------------------------------------------#
# This step will spend some time to load the data        #
#--------------------------------------------------------#
st = time.time()
train_imgs, classes_count, class_mapping = get_data(train_path)
print()
print('Spend %0.2f mins to load the data' % ((time.time()-st)/60))
########################################################################################################

########################################################################################################
if 'bg' not in classes_count:
	classes_count['bg'] = 0
	class_mapping['bg'] = len(class_mapping)
# e.g.
#    classes_count: {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745, 'bg': 0}
#    class_mapping: {'Person': 0, 'Car': 1, 'Mobile phone': 2, 'bg': 3}
C.class_mapping = class_mapping

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print(class_mapping)
########################################################################################################

########################################################################################################
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (VGG here, can be Resnet50, Inception, etc)
shared_layers = model(img_input)
########################################################################################################

########################################################################################################
# define the RPN, built on the base layers
num_classes = len(classes_count)-1
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
rpn = rpn_layer(shared_layers, num_anchors)

classifier = classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count))

model_rpn = Model(img_input, rpn)
model_classifier = Model([img_input, roi_input], classifier)

# Because the google colab can only run the session several hours one time (then you need to connect again), 
# we need to save the model and load the model to continue training
if not os.path.isfile(C.model_rpn_path):
    #If this is the begin of the training, load the pre-traind base network such as vgg-16
    try:
        print('This is the first time of your training')
        print('loading weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(os.path.join(base_path, 'resnet50.h5'), by_name=True)
        model_classifier.load_weights(os.path.join(base_path, 'resnet50.h5'), by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            https://github.com/fchollet/keras/tree/master/keras/applications')
    
    # Create the record.csv file to record losses, acc and mAP
    record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
else:
    # If this is a continued training, load the trained model from before
    print('Continue training based on previous trained model')
    print('Loading weights from {}'.format(C.model_path))
    model_rpn = tf.keras.models.load_model(C.model_rpn_path, custom_objects={'rpn_loss_cls_fixed_num': rpn_loss_cls_fixed_num,
                                                                         'rpn_loss_regr_fixed_num': rpn_loss_regr_fixed_num})
    model_classifier = tf.keras.models.load_model(C.model_cls_path, custom_objects={'RoiPoolingConv': RoiPoolingConv,
                                                                                'class_loss_cls': class_loss_cls,
                                                                                'class_loss_regr': class_loss_regr,
                                                                                'class_loss_regr_fixed_num': class_loss_regr_fixed_num
                                                                                })
    
    # Load the records
    record_df = pd.read_csv(record_path)

    r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
    r_class_acc = record_df['class_acc']
    r_loss_rpn_cls = record_df['loss_rpn_cls']
    r_loss_rpn_regr = record_df['loss_rpn_regr']
    r_loss_class_cls = record_df['loss_class_cls']
    r_loss_class_regr = record_df['loss_class_regr']
    r_curr_loss = record_df['curr_loss']
    r_elapsed_time = record_df['elapsed_time']
    r_mAP = record_df['mAP']

    print('Already train %dK batches'% (len(record_df)))
########################################################################################################

########################################################################################################
# Model prediction
def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(300)
	(height,width,_) = img.shape
		
	if width &lt;= height:
		ratio = img_min_side/width
		new_height = 400
		new_width = 300
	else:
		ratio = img_min_side/height
		new_width = 400
		new_height = 300
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio	


def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = preprocess_input(img)

	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):

	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2 ,real_y2)
########################################################################################################

########################################################################################################
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
########################################################################################################

########################################################################################################
#test_path = os.path.join(annotation_path, 'annotation_test.txt')

test_base_path =  '../input/2019-3rd-ml-month-with-kakr/test/'

test_imgs = os.listdir(test_base_path)

results = []

imgs_path = ['test_01360.jpg', 'test_02664.jpg', 'test_03866.jpg', 'test_04494.jpg', 'test_05546.jpg', 'test_04747.jpg', 'test_04958.jpg']
#for i in range(10):
    #idx = np.random.randint(len(test_imgs))
    #imgs_path.append(test_imgs[idx])

all_imgs = []

classes = {}
########################################################################################################

########################################################################################################
img = plt.imread(filepath)

X, ratio = format_img(img, C)



# get output layer Y1, Y2 from the RPN
# Y1: y_rpn_cls
# Y2: y_rpn_regr
[Y1, Y2] = model_rpn.predict(X)

# Get bboxes by applying NMS 
# R.shape = (300, 4)
R = rpn_to_roi(Y1, Y2, C, overlap_thresh=0.8)

# convert from (x1,y1,x2,y2) to (x,y,w,h)
R[:, 2] -= R[:, 0]
R[:, 3] -= R[:, 1]

# apply the spatial pyramid pooling to the proposed regions
bboxes = {}
probs = {}

for jk in range(R.shape[0]//C.num_rois + 1):
    ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
    if ROIs.shape[1] == 0:
        break

    if jk == R.shape[0]//C.num_rois:
        #pad R
        curr_shape = ROIs.shape
        target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
        ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
        ROIs_padded[:, :curr_shape[1], :] = ROIs
        ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
        ROIs = ROIs_padded

    [P_cls, P_regr] = model_classifier.predict([X, ROIs])

    # Calculate bboxes coordinates on resized image
    for ii in range(P_cls.shape[1]):
        #Ignore 'bg' class
        if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
           continue

        cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

        if cls_name not in bboxes:
            bboxes[cls_name] = []
            probs[cls_name] = []

        (x, y, w, h) = ROIs[0, ii, :]

        cls_num = np.argmax(P_cls[0, ii, :])
        try:
            (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
            tx /= C.classifier_regr_std[0]
            ty /= C.classifier_regr_std[1]
            tw /= C.classifier_regr_std[2]
            th /= C.classifier_regr_std[3]
            x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
        except:
            pass
        bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
        probs[cls_name].append(np.max(P_cls[0, ii, :]))

all_dets = []

for key in bboxes:
    bbox = np.array(bboxes[key])

    new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.7)
    for jk in range(new_boxes.shape[0]):
        (x1, y1, x2, y2) = new_boxes[jk,:]

        # Calculate real coordinates on original image
        (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

        cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

        textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
        all_dets.append((key,100*new_probs[jk]))

        (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
        textOrg = (real_x1, real_y1-0)

        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
        cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

print('Elapsed time = {}'.format(time.time() - st))
print(all_dets)
plt.figure(figsize=(10,10))
plt.grid()
plt.imshow(img)
plt.show()
########################################################################################################