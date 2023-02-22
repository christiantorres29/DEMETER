import matplotlib.pyplot as plt
from skimage.morphology import disk
import cv2
import time

img=cv2.imread("./c3.png")[:,:,2]
kernel = disk(3)

start = time.time()

denoised= cv2.medianBlur(img,19)
denoised = cv2.GaussianBlur(denoised, (15, 15), 1)

gradient =cv2.Laplacian(denoised,cv2.CV_8U)
gradient= cv2.medianBlur(gradient,5)
gradient = cv2.GaussianBlur(gradient,(1,1),0)
ret, markers = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY)

markers = 255-cv2.dilate(markers,kernel,iterations =2)
ret, markers_ = cv2.connectedComponents(markers)

# process the watershed
k=cv2.merge((img,img,img))
labels=cv2.watershed(k,markers_)

end = time.time()
print(end-start) 

# display results
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                         sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(denoised, cmap=plt.cm.gray)
ax[0].set_title("Denoised")

ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
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

metod='laplace'

plt.imsave('./denoised_'+str(metod)+'.png',denoised,cmap=plt.cm.gray)
plt.imsave('./gradient_'+str(metod)+'.png',gradient,cmap=plt.cm.nipy_spectral)
plt.imsave('./markers_'+str(metod)+'.png',markers,cmap=plt.cm.nipy_spectral)
plt.imsave('./segmented_'+str(metod)+'.png',labels,cmap=plt.cm.nipy_spectral)