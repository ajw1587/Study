# https://thispointer.com/python-find-unique-values-in-a-numpy-array-with-frequency-indices-numpy-unique/
import numpy

# Find Unique Values from a Numpy Array
arr = numpy.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18])
uniqueValues = numpy.unique(arr)
print('Original Numpy Array : ' , arr)
print('Unique Values : ',uniqueValues)
print('\n')

# Find Unique Values & their first index position from a Numpy Array
uniqueValues, indicesList = numpy.unique(arr, return_index=True)
print('Unique Values : ', uniqueValues)
print('Indices of Unique Values : ', indicesList)
print('\n')

# Get Unique Values & their frequency count from a Numpy Array
uniqueValues, occurCount = numpy.unique(arr, return_counts=True)
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)

# Get Unique Values , frequency count & index position from a Numpy Array
uniqueValues , indicesList, occurCount= numpy.unique(arr, return_index=True, return_counts=True)