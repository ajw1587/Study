import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# img = cv.imread('F:\Team Project\Image_data\01.jpg')
img = cv.imread('F:/Team Project/Image_data/01.jpg')
# cv.imshow("img", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

img = np.array(img)
print(img.shape)

def Text_Line(img, limt=200):
    image  = np.array(img, dtype=np.float)
    pos_limt = int(limt)
    y = []
    for i, im in enumerate (image):
        c = np.argmin(im)
        if c > pos_limt :
            c= 255
        else :
            c=0
        y.append(c)
    return y

def Line_edge(Texlin, limt=220,):
    c = []
    pos_limt = limt
    neg_limt = -limt
    for i in range(len(Texlin)):
        if i == (len(Texlin)-1):
            break
        tmp =  Texlin[i+1] - Texlin[i] / 1
        if tmp  < neg_limt :
            tmp = -255
        elif tmp  > pos_limt:
            tmp = 255
        else :
            tmp =0
        c.append(tmp)
    c.append(0)
    return c

def edge_index(edge_vector):
    returns = []
    for index, y  in enumerate(edge_vector):
        if y > 0:
            returns.append(index)
        elif y < 0:
            returns.append(index)
    returns = np.array(returns)
    returns = returns.reshape(-1,2)
    return returns

            
word_line = Text_Line(img)
word_line_edge = Line_edge(word_line)
y_location = edge_index(word_line_edge)

# print(y_location.shape) #(22,)
# print(y_location) # [ 24  46  47  48  61  62  63  85 100 125 139 164 178 202 217 239 256 279 295 318 333 357]
for i, yl in enumerate(y_location):
    img_split = img[yl[0]:yl[1]]
    cv.imshow("img", img_split)
    cv.waitKey(0)
    cv.destroyAllWindows()

plt.figure(figsize=(10,4))
plt.subplot(2,1,1)
plt.plot(word_line)


plt.subplot(2,1,2)
plt.plot(word_line_edge)
plt.show()
