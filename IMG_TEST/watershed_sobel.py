import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
import time
import cv2

img=cv2.imread("./c3.png")[:,:,2]
kernel = disk(1)

start = time.time()

# denoise image
denoised = cv2.medianBlur(img,5)

gX = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
gY = cv2.Sobel(denoised, cv2.CV_64F, 0, 1,ksize=3)
# compute the gradient magnitude and orientation
gradient = np.sqrt((gX ** 2) + (gY ** 2)).astype(np.uint8)

gradient = cv2.GaussianBlur(gradient,(9,9),0)#9 9,21 3, 9 17, 9 3
ret, markers_ = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#markers_=cv2.threshold(gradient,26,255,cv2.THRESH_BINARY)[1]
markers_=markers_.astype(np.uint8)

markers_ = 255-cv2.dilate(markers_,kernel,iterations = 5)
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

# metod='sobel'

# plt.imsave('./denoised_'+str(metod)+'.png',denoised,cmap=plt.cm.gray)
# plt.imsave('./gradient_'+str(metod)+'.png',gradient,cmap=plt.cm.nipy_spectral)
# plt.imsave('./markers_'+str(metod)+'.png',markers_,cmap=plt.cm.nipy_spectral)
# plt.imsave('./segmented_'+str(metod)+'.png',labels,cmap=plt.cm.nipy_spectral)