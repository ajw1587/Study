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

def generate_annotation_xml(width, height, depth, name1, x1, y1, name2, x2, y2, name3, x3, y3):
    

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

    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    # labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',
    #                      encoding='utf-8')

    total_count = 0
    prev_count = 0
    text_size = 20

    label_list = []
    path_list = []
    for character in labels:
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        for font in fonts:
            
            # 위치 생성
            # while True:
            #     x1 = random.randint(0 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))
            #     y1 = random.randint(0 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))

            #     x2 = random.randint(0 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))
            #     y2 = random.randint(0 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))

            #     x3 = random.randint(0 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))
            #     y3 = random.randint(0 + math.ceil(text_size/2), 500 - math.ceil(text_size/2))

            #     dis1 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            #     dis2 = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)
            #     dis3 = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)

            #     if dis1 < text_size or dis2 < text_size or dis3 < text_size:
            #         continue
            #     else:
            #         break
            x1, y1, x2, y2, x3, y3 = generate_hangul_location(0, 500, text_size)

            total_count += 1
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
            font = ImageFont.truetype(font, text_size)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)

            # drawing 1
            drawing.text(
                (x1, y1), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0),
                font=font
            )
            # drawing 2
            drawing.text(
                (x2, y2), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0),
                font=font
            )
            # drawing 3
            drawing.text(
                (x3, y3), # ((IMAGE_WIDTH)/2, (IMAGE_HEIGHT)/2),
                character,
                fill=(0),
                font=font
            )

            file_string = 'hangul_{}.jpeg'.format(total_count)
            file_path = os.path.join(image_dir, file_string)
            image.save(file_path, 'JPEG')


            label_list.append(character)
            path_list.append(file_path)
            # labels_csv.write(u'{},{}\n'.format(file_path, character))

            for i in range(DISTORTION_COUNT):
                total_count += 1
                file_string = 'hangul_{}.jpeg'.format(total_count)
                file_path = os.path.join(image_dir, file_string)
                arr = numpy.array(image)

                distorted_array = elastic_distort(
                    arr, alpha=random.randint(30, 36),
                    sigma=random.randint(5, 6)
                )
                distorted_image = Image.fromarray(distorted_array)
                distorted_image.save(file_path, 'JPEG')


                label_list.append(character)
                path_list.append(file_path)

                # labels_csv.write(u'{},{}\n'.format(file_path, character))
    label_list = numpy.array(label_list)
    path_list = numpy.array(path_list)

    label_list = label_list.reshape(-1, 1)
    path_list = path_list.reshape(-1, 1)

    final_list = numpy.concatenate((path_list, label_list), axis = 1)

    final_list = pd.DataFrame(final_list)
    final_list.to_csv('F:/Team Project/OCR/02_Image_to_Text_model/test_data/test-labels-map.csv'
                      , index_label = False
                      , header = False)


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