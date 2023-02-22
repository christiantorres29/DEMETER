import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
import skimage.filters
import sklearn.metrics

# Turn on interactive mode. Turn off with plt.ioff()
plt.ion()

grayscale = cv2.imread('c3.png')[:,:,1]
grayscale = 255 - grayscale
groundtruth = scipy.misc.imread('groundtruth.png')
plt.subplot(1, 3, 1)
plt.imshow(255 - grayscale, cmap='gray')
plt.title('grayscale')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(grayscale, cmap='gray')
plt.title('inverted grayscale')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(groundtruth, cmap='gray')
plt.title('groundtruth binary')
plt.axis('off')