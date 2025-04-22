# Contours are different from edges in that they are continuous lines
# that bound a shape. They are useful for shape analysis and object
# detection and recognition.
#

import cv2 as cv
import numpy as np

img = cv.imread('Resources/Photos/cats.jpg')
#cv.imshow('Cats', img)

blank = np.zeros(img.shape, dtype='uint8')
#cv.imshow('Blank Image', blank)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

# blur before contrours to reduce noise
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

canny = cv.Canny(blur, 125, 175)
#cv.imshow('Canny Edges', canny)

# Binarize the image (black under 125, white over 125)
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

# Find contours
# RETR_LIST: retrieves all contours
# CHAIN_APPROX_NONE: stores all the boundary points
# CHAIN_APPROX_SIMPLE: stores only the end points of the contours
# CHAIN_APPROX_TC89_L1: stores only the end points of the contours
# CHAIN_APPROX_TC89_KCOS: stores only the end points of the contours
# contours: list of contours
# hierarchy: list of contours' hierarchy
contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

# Draw contours
# -1: draw all contours
# 0: draw contour 0
# 1: draw contour 1
# 2: draw contour 2
# ...
cv.drawContours(blank, contours, -1, (0,0,255), 1)
cv.imshow('Contours Drawn', blank)




cv.waitKey(0)