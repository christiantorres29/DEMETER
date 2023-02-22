import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
import time
import cv2

#read image
img=cv2.imread("./d50.png",0)

#create kernel and mask for blur and filtering
kernel = disk(6) #dik 6 rad 150 gaussblur 121

# create circle mask
radius = 150
#mask = np.ones_like(img)
mask = np.zeros((480,640))
cy = mask.shape[0] // 2
cx = mask.shape[1] // 2
cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
mask=255-mask
mask=cv2.GaussianBlur(mask.copy(),(31,181),0)
mask=cv2.merge((mask.copy(),mask.copy()))

# denoise image
start=time.time()

denoised = cv2.medianBlur(img,1)

dft = cv2.dft(np.float32(denoised), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
gradient = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
gradient = cv2.normalize(gradient,0,255,cv2.NORM_MINMAX)
gradient = cv2.threshold(gradient,0.51,255,cv2.THRESH_BINARY)[1].astype(np.uint8)

markers = 255-cv2.dilate(gradient,kernel,iterations = 1)
ret, markers_ = cv2.connectedComponents(markers)

# process the watershed
k=cv2.merge((img,img,img))
labels=cv2.watershed(k,markers_)

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

ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
ax[2].set_title("Markers")

ax[3].imshow(img, cmap=plt.cm.gray)
ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.5)
ax[3].set_title("Segmented")

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()

metod='fft'

plt.imsave('./denoised_'+str(metod)+'.png',denoised,cmap=plt.cm.gray)
plt.imsave('./gradient_'+str(metod)+'.png',gradient,cmap=plt.cm.nipy_spectral)
plt.imsave('./markers_'+str(metod)+'.png',markers,cmap=plt.cm.nipy_spectral)
plt.imsave('./segmented_'+str(metod)+'.png',labels,cmap=plt.cm.nipy_spectral)