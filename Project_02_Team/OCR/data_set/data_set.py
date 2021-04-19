import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def load_shapes(self, count, height, width):
    """Generate the requested number of synthetic images. 
             count: number of images to generate. number
             height, width: the size of the generated images. size
    """
    # Add classes
    # self.add_class("shapes", 1, "square")
    # self.add_class("shapes", 2, "circle")
    # self.add_class("shapes", 3, "triangle")
    # self.add_class("shapes", 0,'BG') # Label 0 defaults to background
    [self.add_class("shapes", i+1, str(i)) for i in range(10)] # 0~9 corresponds to label 1~10; label 0 defaults to background

    # Add images
    for i in range(count):
        comb_image, mask, class_ids=self.random_comb(mnist,height,width)

        # The input picture is 3 bands by default
        comb_images=np.zeros([height,width,3],np.float32)
        comb_images[:,:,0]=comb_image
        comb_images[:, :, 1] = comb_image
        comb_images[:, :, 2] = comb_image
        comb_image=comb_images # [128,128,3] Convert to 3 band

        mask=np.asarray(mask).transpose([1, 2, 0]) # mask shape [128,128,16]
        self.add_image("shapes", image_id=i, path=None,
                       width=width, height=height,
                       image=comb_image,mask=mask,class_ids=class_ids)


def load_image(self, image_id):
    info = self.image_info[image_id]
    image=info['image']
    return image


def load_mask(self, image_id):
    """Generate instance masks for shapes of the given image ID.
    The mask corresponds to the class name """

    info = self.image_info[image_id]
    mask=info['mask']
    class_ids=info['class_ids']
    return mask, class_ids.astype(np.int32)