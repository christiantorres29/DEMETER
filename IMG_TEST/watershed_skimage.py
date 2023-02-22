from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
import cv2
import time

img = cv2.imread("c3.png",0)#img_as_ubyte(data.eagle())

start = time.time()
# denoise image
denoised = rank.median(img, disk(5))

# local gradient (disk(2) is used to keep edges thin)
gradient = rank.gradient(denoised, disk(6))

# find continuous region (low gradient -
# where less than 10 for this image) --> markers
# disk(5) is used here to get a more smooth image
markers_ = gradient < 15
markers = ndi.label(markers_)[0]

# process the watershed
labels = watershed(gradient, markers)

end=time.time()
print(end-start)
# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title("Original")

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

metod='skymage'

"""
plt.imsave('./denoised_'+str(metod)+'.png',denoised,cmap=plt.cm.gray)
plt.imsave('./gradient_'+str(metod)+'.png',gradient,cmap=plt.cm.nipy_spectral)
plt.imsave('./markers_'+str(metod)+'.png',markers_,cmap=plt.cm.nipy_spectral)
plt.imsave('./segmented_'+str(metod)+'.png',labels,cmap=plt.cm.nipy_spectral)
"""