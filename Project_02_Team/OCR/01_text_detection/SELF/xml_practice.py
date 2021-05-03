# https://ballentain.tistory.com/8

from xml.etree.ElementTree import Element, SubElement, ElementTree

filename = 'annotation'
width = 512
height = 366
depth = 3

root = Element('annotations')
SubElement(root, 'folder').text = 'images'

SubElement(root, 'filename').text = 'example'

size = SubElement(root, 'size')
SubElement(size, 'width').text = '1'
SubElement(size, 'height').text = '2'
SubElement(size, 'depth').text = '3'

SubElement(root, 'segmented').text = '0'

# object 1
object = SubElement(root, 'object')
SubElement(object, 'name').text = 'without_mask'
SubElement(object, 'pose').text = 'Unspecified'
SubElement(object, 'truncated').text = '0'
SubElement(object, 'occluded').text = '0'
SubElement(object, 'difficult').text = '0'

bnd = SubElement(object, 'bndbox')
SubElement(bnd, 'xmin').text = '10'
SubElement(bnd, 'ymin').text = '10'
SubElement(bnd, 'xmax').text = '20'
SubElement(bnd, 'ymax').text = '20'

# object 2
object = SubElement(root, 'object')
SubElement(object, 'name').text = 'without_mask'
SubElement(object, 'pose').text = 'Unspecified'
SubElement(object, 'truncated').text = '0'
SubElement(object, 'occluded').text = '0'
SubElement(object, 'difficult').text = '0'

bnd = SubElement(object, 'bndbox')
SubElement(bnd, 'xmin').text = '10'
SubElement(bnd, 'ymin').text = '10'
SubElement(bnd, 'xmax').text = '20'
SubElement(bnd, 'ymax').text = '20'

# object 3
object = SubElement(root, 'object')
SubElement(object, 'name').text = 'without_mask'
SubElement(object, 'pose').text = 'Unspecified'
SubElement(object, 'truncated').text = '0'
SubElement(object, 'occluded').text = '0'
SubElement(object, 'difficult').text = '0'

bnd = SubElement(object, 'bndbox')
SubElement(bnd, 'xmin').text = '10'
SubElement(bnd, 'ymin').text = '10'
SubElement(bnd, 'xmax').text = '20'
SubElement(bnd, 'ymax').text = '20'

tree = ElementTree(root)
tree.write('F:/xmltest/' + filename + '.xml')