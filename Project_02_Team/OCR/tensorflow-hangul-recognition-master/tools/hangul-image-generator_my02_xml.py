#!/usr/bin/env python

import argparse
import glob
import io
import os
import random
import math

import numpy
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from xml.etree.ElementTree import Element, SubElement, ElementTree


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
# '../labels/2350-common-hangul.txt'
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, 'C:/Study/Project_02_Team/OCR/tensorflow-hangul-recognition-master/labels/2350-common-hangul-3.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'C:/Study/Project_02_Team/OCR/tensorflow-hangul-recognition-master/fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'F:/Team Project/OCR/02_Image_to_Text_model/test_data')
# C:\Users\Admin\Desktop\image-data
# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 1

# Width and height of the resulting image.
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
DEPTH = 3

def generate_hangul_location(start, end, text_size):
    while True:
        x1 = random.randint(start + math.ceil(text_size/2), end - math.ceil(text_size/2))
        y1 = random.randint(start + math.ceil(text_size/2), end - math.ceil(text_size/2))

        x2 = random.randint(start + math.ceil(text_size/2), end - math.ceil(text_size/2))
        y2 = random.randint(start + math.ceil(text_size/2), end - math.ceil(text_size/2))

        x3 = random.randint(start + math.ceil(text_size/2), end - math.ceil(text_size/2))
        y3 = random.randint(start + math.ceil(text_size/2), end - math.ceil(text_size/2))

        dis1 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        dis2 = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        dis3 = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)

        if dis1 < text_size or dis2 < text_size or dis3 < text_size:
            continue
        else:
            return x1, y1, x2, y2, x3, y3

def generate_annotation_xml(filename, path, width, height, depth, text_size, name, x1, y1, x2, y2, x3, y3):
    root = Element('annotation')
    SubElement(root, 'folder').text = 'images'

    SubElement(root, 'filename').text = filename + '.png'

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(width)
    SubElement(size, 'height').text = str(height)
    SubElement(size, 'depth').text = str(depth)

    SubElement(root, 'segmented').text = '0'

    # object 1
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = name
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0'
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    SubElement(bnd, 'xmin').text = str(x1 - text_size/2)
    SubElement(bnd, 'ymin').text = str(y1 - text_size/2)
    SubElement(bnd, 'xmax').text = str(x1 + text_size/2)
    SubElement(bnd, 'ymax').text = str(y1 + text_size/2)

    # object 2
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = name
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0'
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    SubElement(bnd, 'xmin').text = str(x2 - text_size/2)
    SubElement(bnd, 'ymin').text = str(y2 - text_size/2)
    SubElement(bnd, 'xmax').text = str(x2 + text_size/2)
    SubElement(bnd, 'ymax').text = str(y2 + text_size/2)

    # object 3
    object = SubElement(root, 'object')
    SubElement(object, 'name').text = name
    SubElement(object, 'pose').text = 'Unspecified'
    SubElement(object, 'truncated').text = '0'
    SubElement(object, 'occluded').text = '0'
    SubElement(object, 'difficult').text = '0'

    bnd = SubElement(object, 'bndbox')
    SubElement(bnd, 'xmin').text = str(x3 - text_size/2)
    SubElement(bnd, 'ymin').text = str(y3 - text_size/2)
    SubElement(bnd, 'xmax').text = str(x3 + text_size/2)
    SubElement(bnd, 'ymax').text = str(y3 + text_size/2)

    tree = ElementTree(root)
    tree.write(path + filename + '.xml')

def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    # image_dir = os.path.join(output_dir, 'hangul-images')

    TRAIN_IMAGE_DIR = os.path.join(output_dir, 'train_hangul-images')
    TEST_IMAGE_DIR = os.path.join(output_dir, 'test_hangul-images')
    if not os.path.exists(TRAIN_IMAGE_DIR):
        os.makedirs(os.path.join(TRAIN_IMAGE_DIR))
    if not os.path.exists(TEST_IMAGE_DIR):
        os.makedirs(os.path.join(TEST_IMAGE_DIR))
    # if not os.path.exists(image_dir):
    #     os.makedirs(os.path.join(image_dir))
    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    # labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
    #                      encoding='utf-8')

    total_count = 0
    prev_count = 0
    text_size = 20
    TRAIN_ANNOTATION_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/test_data/train_annotation/'
    TEST_ANNOTATION_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/test_data/test_annotation/'
    if not os.path.exists(TRAIN_ANNOTATION_PATH):
        os.makedirs(os.path.join(TRAIN_ANNOTATION_PATH))
    if not os.path.exists(TEST_ANNOTATION_PATH):
        os.makedirs(os.path.join(TEST_ANNOTATION_PATH))

    label_list = []
    path_list = []
    for character in labels:
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        for font in fonts:
            
            # 위치 생성
            x1, y1, x2, y2, x3, y3 = generate_hangul_location(0, 500, text_size)

            total_count += 1
            image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=(255, 255, 255)) # 'L' -> 'RGB'
            font = ImageFont.truetype(font, text_size)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)

            # drawing 1
            drawing.text(
                (x1, y1), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font
            )
            # drawing 2
            drawing.text(
                (x2, y2), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font
            )
            # drawing 3
            drawing.text(
                (x3, y3), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0, 0, 0),
                font=font
            )

            file_string = 'hangul_{}.png'.format(total_count)
            file_path = os.path.join(TEST_IMAGE_DIR, file_string)
            # shape = numpy.array(image)
            # print('shape.shape: ', shape.shape)
            image.save(file_path, 'PNG')

            annotation_name = 'hangul_{}'.format(total_count)
            generate_annotation_xml(annotation_name, TEST_ANNOTATION_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH, text_size, character, x1, y1, x2, y2, x3, y3)

            # label_list.append(character)
            # path_list.append(file_path)

            # labels_csv.write(u'{},{}\n'.format(file_path, character))

            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = 'hangul_{}.png'.format(total_count)
                file_path = os.path.join(TRAIN_IMAGE_DIR, file_string)
                arr = numpy.array(image)
                # print('arr.shape: ', arr.shape)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                distorted_image = Image.fromarray(distorted_array)
                distorted_image.save(file_path, 'PNG')
                generate_annotation_xml(annotation_name, TRAIN_ANNOTATION_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, DEPTH, text_size, character, x1, y1, x2, y2, x3, y3)

                # label_list.append(character)
                # path_list.append(file_path)

                # labels_csv.write(u'{},{}\n'.format(file_path, character))

    # label_list = numpy.array(label_list)
    # path_list = numpy.array(path_list)

    # label_list = label_list.reshape(-1, 1)
    # path_list = path_list.reshape(-1, 1)

    # final_list = numpy.concatenate((path_list, label_list), axis = 1)

    # final_list = pd.DataFrame(final_list)
    # final_list.to_csv('F:/Team Project/OCR/02_Image_to_Text_model/test_data/test-labels-map.csv'
    #                   , index_label = False
    #                   , header = False)


    print('Finished generating {} images.'.format(total_count))
    # labels_csv.close()


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = numpy.random.RandomState(None)
    shape = image.shape
    print('image.shape: ', image.shape)
    print('shape: ', shape)

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)

# DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
#                                   '../labels/2350-common-hangul.txt')
# DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../fonts')
# DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'C:/Users/Admin/Desktop/image-data')
# # C:\Users\Admin\Desktop\image-data
# # Number of random distortion images to generate per font and character.
# DISTORTION_COUNT = 3

# # Width and height of the resulting image.
# IMAGE_WIDTH = 64
# IMAGE_HEIGHT = 64