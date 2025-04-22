import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('Resources/Photos/cats.jpg')
cv.imshow('Cats', img)

# Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Grayscale histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
cv.imshow('Gray Histogram', gray_hist)

# Color histogram
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins') # intensity levels
plt.ylabel('# of pixels') # number of pixels in each bin
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
plt.show()

# Masking
blank = np.zeros(img.shape[:2], dtype='uint8') # same size is important
cv.imshow('Blank Image', blank)

mask = cv.circle(blank, (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask', mask)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked)

# Masked histogram
masked_hist = cv.calcHist([img], [0], mask, [256], [0, 256])
cv.imshow('Masked Histogram', masked_hist)


cv.waitKey(0)
