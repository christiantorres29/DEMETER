import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
import cv2
import time

img=cv2.imread("./furrows.png")
r=cv2.equalizeHist(img[:,:,0])
g=cv2.equalizeHist(img[:,:,1])
b=cv2.equalizeHist(img[:,:,2])
kernel = disk(11)
kernel2 =disk(1)

#green = np.uint8([0,255,0 ])
#hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)

#hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# define range of green color in HSV
#lower_green = np.array(green*0.7)
#upper_green = np.array(green*1.3)
# Threshold the HSV image to get only green colors
#mask = cv2.inRange(hsv, lower_green, upper_green)
# Bitwise-AND mask and original image
#res = cv.bitwise_and(frame,frame, mask= mask)


img=g

start = time.time()

hpf = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,61,30)
#hpf = cv2.adaptiveThreshold(hpf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,61,30)

denoised = hpf#cv2.medianBlur(hpf,5)

#gX = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
#gY = cv2.Sobel(denoised, cv2.CV_64F, 0, 1,ksize=3)
# compute the gradient magnitude and orientation
#gradient = #np.sqrt((gX ** 2) + (gY ** 2)).astype(np.uint8)

#hpf = cv2.GaussianBlur(gradient,(9,9),0)#9 9,21 3, 9 17, 9 3

_,denoised = cv2.threshold(denoised,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


markers = cv2.dilate(denoised,kernel,iterations = 1)
markers = 255 - cv2.erode(markers,kernel2,iterations=1)
ret, markers_ = cv2.connectedComponents(markers)

# process the watershed
k=cv2.merge((img,img,img))
labels=cv2.watershed(k,markers_)

end = time.time()
print((end-start)) 

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title("Original")

ax[1].imshow(hpf, cmap=plt.cm.nipy_spectral)
ax[1].set_title("Image under HPF")

ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
ax[2].set_title("Markers")

ax[3].imshow(img, cmap=plt.cm.gray)
ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()

metod='hpf'

"""
plt.imsave('./gradient_'+str(metod)+'.png',hpf,cmap=plt.cm.nipy_spectral)
plt.imsave('./markers_'+str(metod)+'.png',markers,cmap=plt.cm.nipy_spectral)
plt.imsave('./segmented_'+str(metod)+'.png',labels,cmap=plt.cm.nipy_spectral)
"""
