import pandas as pd
import numpy as np

# a = np.array(["'", '"', ','])
# df = pd.DataFrame(a)
# df.columns = ['test']

a = pd.read_csv('F:/Team Project/OCR/Image_to_Text_model/image-data/my_hangul_images/labels-map.csv')
a = np.array(a)
print(a.shape)
print(a[328500:, :])

# ('F:/Team Project/OCR/Image_to_Text_model/image-data/my_hangul_images')

# b = pd.read_csv('')