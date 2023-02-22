
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
import cv2
import time

img=cv2.imread("./d50.png",0)
kernel = disk(0)
start = time.time()

# denoise image
denoised = cv2.GaussianBlur(img,(11, 11),0)## 7 7  , 9 15, 9 19

gX = cv2.Scharr(denoised, cv2.CV_64F, 1, 0)
gY = cv2.Scharr(denoised, cv2.CV_64F, 0, 1)
# compute the gradient magnitude and orientation
gradient = np.sqrt((gX ** 2) + (gY ** 2)).astype(np.uint8)

gradient = cv2.GaussianBlur(gradient,(31,31),0)## 31 31, 31 31, 31 31
ret, markers_ = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

ret, markers = cv2.connectedComponents(markers_)

# process the watershed
k=cv2.merge((img,img,img))
labels=cv2.watershed(k,markers.copy())
end=time.time()
print(end-start) 

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(denoised, cmap=plt.cm.gray)
ax[0].set_title("Denoised")

ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
ax[1].set_title("Local Gradient")

ax[2].imshow(markers_, cmap=plt.cm.nipy_spectral)
ax[2].set_title("Markers")

ax[3].imshow(img, cmap=plt.cm.gray)
ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()

metod='scharr'

plt.imsave('./denoised_'+str(metod)+'.png',denoised,cmap=plt.cm.gray)
plt.imsave('./gradient_'+str(metod)+'.png',gradient,cmap=plt.cm.nipy_spectral)
plt.imsave('./markers_'+str(metod)+'.png',markers_,cmap=plt.cm.nipy_spectral)
plt.imsave('./segmented_'+str(metod)+'.png',labels,cmap=plt.cm.nipy_spectral)