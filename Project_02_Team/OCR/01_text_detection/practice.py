from xml.etree import ElementTree as ET

# <?xml version="1.0"?>
# -<annotation verified="yes">
# <folder>images</folder>
# <filename>raccoon-1.jpg</filename>
# <path>/Users/datitran/Desktop/raccoon/images/raccoon-1.jpg</path>
# -<source>
# <database>Unknown</database>
# </source>
# -<size>
# <width>650</width>
# <height>417</height>
# <depth>3</depth>
# </size>
# <segmented>0</segmented>
# -<object>
# <name>raccoon</name>
# <pose>Unspecified</pose>
# <truncated>0</truncated>
# <difficult>0</difficult>
# -<bndbox>
# <xmin>81</xmin>
# <ymin>88</ymin>
# <xmax>522</xmax>
# <ymax>408</ymax>
# </bndbox>
# </object>
# </annotation>

tree = ET.parse('F:/Team Project/OCR/01_Text_detection/Image_data/raccoon/annotations/raccoon-12.xml')
root = tree.getroot()
# print(tree)
# print(tree.find('size').find('width').text)
# print(tree.find('folder').text)
# print(tree.find('object').find('name'))
# print(tree.find('object').)
# print(tree.find('object').find('bndbox').find('xmin').text)
# print(tree.find('object').find('bndbox').find('ymin').text)
# print(tree.find('object').find('bndbox').find('xmax').text)
# print(tree.find('object').find('bndbox').find('ymax').text)

