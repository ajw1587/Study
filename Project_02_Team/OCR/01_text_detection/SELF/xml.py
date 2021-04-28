from xml.etree.ElementTree import Element, SubElement, ElementTree

filename = 'annotation'
width = 512
height = 366
depth = 3



tree = ElementTree(root)
tree.write('F:/xmltest' + filename + '.xml')