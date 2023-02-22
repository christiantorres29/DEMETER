import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
import cv2
import time

img=cv2.imread("./d50.png",0)
kernel = disk(4)

start = time.time()
denoised= cv2.medianBlur(img,3)

edges=cv2.Canny(denoised,45,50)
markers = 255 - cv2.dilate(edges,kernel,iterations = 1)
ret, markers_ = cv2.connectedComponents(markers)

# process the watershed
k=cv2.merge((img,img,img))
labels=cv2.watershed(k, markers_)

end = time.time()
print((end-start)) 

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title("Original")

ax[1].imshow(edges, cmap=plt.cm.nipy_spectral)
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

metod='canny'


plt.imsave('./gradient_'+str(metod)+'.png',edges,cmap=plt.cm.nipy_spectral)
plt.imsave('./markers_'+str(metod)+'.png',markers,cmap=plt.cm.nipy_spectral)
plt.imsave('./segmented_'+str(metod)+'.png',labels,cmap=plt.cm.nipy_spectral)