import numpy as np
import pandas as pd

LABEL_PATH = 'F:/Team Project/OCR/02_Image_to_Text_model/image-data/my_hangul_images/labels-map.csv'

y_data = pd.read_csv(LABEL_PATH, encoding = 'utf-8')

