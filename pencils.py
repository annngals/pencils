# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:33:09 2020

@author: Anna Galsanova
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import threshold_triangle


def toGray(image):
    return (0.2989 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]).astype("uint8")

def binarisation(image, limit_min, limit_max):
    B = image.copy()
    B[B < limit_min] = 0
    B[B >= limit_max] = 0
    B[B > 0] = 1
    return B

def circularity(region, label = 1):
    return (region.perimeter**2) /region.area

pic = 1
pencils = 0

for pic in range(1, 13):
    image = plt.imread("pencils/img ("+str(pic)+").jpg")
    gray = toGray(image)
    
    thresh = threshold_triangle(gray)
    binary = binarisation(gray, 0, thresh)
    
    binary = morphology.binary_dilation(binary, iterations = 1)
    
    labeled = label(binary)
    
    areas = []
    
    for region in regionprops(labeled):
        areas.append(region.area)
    
    for region in regionprops(labeled):
        if region.area < np.mean(areas):
            labeled[labeled == region.label] = 0
        bbox = region.bbox
        if bbox[0] == 0 or bbox[1] == 0:
            labeled[labeled == region.label] = 0
    
    labeled[labeled > 0] = 1
    labeled = label(labeled)
    
    n = 1
    pen = 0
    
    for region in regionprops(labeled):
        if (((circularity(region, n) > 100) and (500000 > region.area > 330000))):
            pen += 1
        n += 1
        
    print(pic, " ", pen)
    pencils += pen
    

print("Pencils: ", pencils)

plt.subplot(131)
plt.imshow(gray, cmap="gray")
plt.subplot(132)
plt.imshow(binary)
plt.subplot(133)
plt.imshow(labeled)

# plt.figure()
# plt.plot(H)

plt.show()
