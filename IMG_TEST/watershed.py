import numpy as np
from skimage.morphology import disk
import cv2

CANNY= 0
LAPLACE= 1
SOBEL= 2
SCHARR= 3
HPF= 4
FFT= 5

# create circle mask for FFT method
radius = 150
mask = np.zeros((480,640))
cy = mask.shape[0] // 2
cx = mask.shape[1] // 2
cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
mask=255-mask
mask=cv2.GaussianBlur(mask.copy(),(31,181),0)
mask=cv2.merge((mask.copy(),mask.copy()))

# create disk kernels for any function
canny_k=disk(4)
fft_k = disk(6) 
laplace_k = disk(3)
sobel_k= disk(1)
hpf_k1 = disk(11)
hpf_k2 =sobel_k.copy()


def canny(img):
    denoised= cv2.medianBlur(img,3)
    edges=cv2.Canny(denoised,45,50)
    markers = 255 - cv2.dilate(edges,canny_k,iterations = 1)
    return markers

def fft(img):
    denoised = cv2.medianBlur(img,1)
    dft = cv2.dft(np.float32(denoised), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    gradient = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    gradient = cv2.normalize(gradient,0,255,cv2.NORM_MINMAX)
    gradient = cv2.threshold(gradient,0.51,255,cv2.THRESH_BINARY)[1].astype(np.uint8)
    markers = 255-cv2.dilate(gradient,fft_k,iterations = 1)

    return markers

def laplace(img):
    denoised= cv2.medianBlur(img,19)
    denoised = cv2.GaussianBlur(denoised, (15, 15), 1)
    gradient =cv2.Laplacian(denoised,cv2.CV_8U)
    gradient= cv2.medianBlur(gradient,5)
    gradient = cv2.GaussianBlur(gradient,(1,1),0)
    _, markers = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY)
    markers = 255-cv2.dilate(markers,laplace_k,iterations =2)
    return markers

def sobel(img):
    # denoise image
    denoised = cv2.medianBlur(img,5)
    gX = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    gY = cv2.Sobel(denoised, cv2.CV_64F, 0, 1,ksize=3)
    # compute the gradient magnitude and orientation
    gradient = np.sqrt((gX ** 2) + (gY ** 2)).astype(np.uint8)
    gradient = cv2.GaussianBlur(gradient,(9,9),0)#9 9,21 3, 9 17, 9 3
    #ret, markers_ = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    markers=cv2.threshold(gradient,26,255,cv2.THRESH_BINARY)[1]
    markers=markers.astype(np.uint8)
    markers = 255-cv2.dilate(markers,sobel_k,iterations = 5)
    return markers

def scharr(img):
    # denoise image
    denoised = cv2.GaussianBlur(img,(11, 11),0)## 7 7  , 9 15, 9 19
    gX = cv2.Scharr(denoised, cv2.CV_64F, 1, 0)
    gY = cv2.Scharr(denoised, cv2.CV_64F, 0, 1)
    # compute the gradient magnitude and orientation
    gradient = np.sqrt((gX ** 2) + (gY ** 2)).astype(np.uint8)
    gradient = cv2.GaussianBlur(gradient,(31,31),0)## 31 31, 31 31, 31 31
    _, markers = cv2.threshold(gradient,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return markers

def hpf(img):
    hpf = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,61,10)
    markers = cv2.dilate(hpf,hpf_k1,iterations = 1)
    markers = 255 - cv2.erode(markers,hpf_k2,iterations=1)
    return markers


def watershed(img,method=CANNY):

    match method:
        case 0:
            markers=canny(img)
        case 1:
            markers=laplace(img)
        case 2:
            markers=sobel(img)
        case 3:
            markers=scharr(img)
        case 4:
            markers=hpf(img)
        case 5:
            markers=fft(img)

    _, markers = cv2.connectedComponents(markers)
    # process the watershed
    k=cv2.merge((img,img,img))
    labels=cv2.watershed(k, markers)
    return labels 