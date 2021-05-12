
from Coordinate import Text_Coordinate
import os
import io
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
path = "./test_data/test_image"
# path = "./test_data/test_img"
# path = "D:\python\pjt_odo/test"
# label_path = "./test_data/test_box"
file = os.listdir(path)




for j, f in enumerate(file):  
    fpath = path + "/" + f
    # labels_txt = io.open(os.path.join(label_path, '{}.text'.format(f)), 'w',
    #                      encoding='utf-8')
    print(fpath)
    image = cv2.imread(fpath)
    # image = tiff.imread(fpath)
    print(image)
    # img_array = np.fromfile(fpath, np.uint8)
    # image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # cv2.imshow("detect_Line", image)
    # cv2.waitKey(0)
    # cv2.destroyWindow("detect")
    # image = cv2.resize(image,(int(image.shape[0]/1.5),int(image.shape[1]/1.5)))

    a = Text_Coordinate(image)
    b = a.bbox()
    print(f)
    for i, word in enumerate(b):
        line = i
        for j in range(len(word)):
            str_index = j
            if b[line,str_index,0] == 1:
                xmin = b[line,str_index,1]
                ymin = b[line,str_index,2]
                xmax = b[line,str_index,3]
                ymax = b[line,str_index,4]
                if xmax-xmin <6:
                    continue
                elif ymax-ymin <6:
                    continue
                green_color = (0,255,0)
                # labels_txt.write(u'{},  {}, {}, {}, {}\n'.format(1, xmin, ymin, xmax, ymax))
                image = cv2.rectangle(image, (xmin, ymin),(xmax,ymax),green_color,2)
    # labels_txt.close()
    image = cv2.resize(image, (int(image.shape[0]*0.25), int(image.shape[1]*0.5)))

    cv2.imshow("detect_Line : {}, Index_Sting : {}".format(line+1,str_index+1), image)
    cv2.waitKey(0)
    cv2.destroyWindow("detect")


# path = "./test_data/test_image"
# label_path = "./test_data/test_box"
# file = os.listdir(path)

# for j, f in enumerate(file):  
#     # fpath = path + "/" + "{}.jpeg".format(j+1)
#     fpath = path + "/" + f
#     label_fpath = label_path + "/" + f[:-3] + "txt"
#     print(fpath)
#     print(label_fpath)
#     labels_txt = open(label_fpath, 'r',
#                          encoding='utf-8')
#     # print(fpath)
#     # print(labels_txt)
#     image = cv2.imread(fpath)
#     # cv2.imshow("image",image)
#     # cv2.waitKey(0)
#     # cv2.destroyWindow("detect")
#     # a = Text_Coordinate(image)
#     # b = a.bbox()
#     for i, word in enumerate(labels_txt):
#         word = word.strip().split(" ")
#         xmin = int(word[1])
#         ymin = abs(int(word[4])-(2570))
#         xmax = int(word[3])
#         ymax = abs(int(word[2])-(2570))
#         green_color = (0,255,0)
#         image = cv2.rectangle(image, (xmin, ymin),(xmax,ymax),green_color,2)
#         # image = cv2.rectangle(image, (0, 0),(200,200),green_color,2)
#     image = cv2.resize(image, (int(image.shape[0]*0.25), int(image.shape[1]*0.5)))
#     cv2.imshow("image",image)
#     cv2.waitKey(0)
#     cv2.destroyWindow("detect")