import numpy as np
import cv2
from skimage.morphology import disk

kernel = disk(5).astype(np.uint8)
# define green color range
low_green = np.array([25, 52, 72])#[25, 52, 72])
high_green = np.array([102, 255, 255])#[102, 255, 255])

def greenbyHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    green_mask = cv2.inRange(hsv, low_green, high_green)
    #green = cv2.bitwise_and(img, img, mask=green_mask)
    green_mask=cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    return green_mask

def greenbyCOM(img):

    r=img[:,:,0]#cv2.equalizeHist(img[:,:,0])
    g=img[:,:,1]#cv2.equalizeHist(img[:,:,1])
    b=img[:,:,2]#cv2.equalizeHist(img[:,:,2])

    T = cv2.add(r.copy(), g.copy(), b.copy())
    try:
        R = r.copy() / T
        G = g.copy() / T
        B = b.copy() / T
    except:
        T[T==0]=1 # eliminar posibles divisiones por cero
        R=r.copy()/T
        G=g.copy()/T
        B=b.copy()/T

    ExG = 2*G-R-B
    ExR = 1.4*R-G
    ExGR = ExG-ExR
    CIVE = 0.441*R-0.811*G + 0.385*G + 18.78745
    RB=(R**0.667 * B**0.333)
    try:
        VEG = G/RB
    except:
        RB[RB==0]=1
        VEG = G/RB

    COM = 0.25*ExG+ 0.30*ExGR+ 0.33*CIVE + 0.12*VEG*R

    COM=cv2.convertScaleAbs(COM, alpha=35)#255/COM.max()

    blur = cv2.GaussianBlur(COM,(3,3),0)
    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh

def greenidfbyHSVDT(img):
    hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H=hsv[:,:,0]*255
    S=hsv[:,:,1]*255
    V=hsv[:,:,2]*255

    H[(H < 50) | (H > 150)] = 0

    H[(H > 49) & (H < 60) & (S > 5) & (S < 50) &(V > 150)] = 0
    t=5 # 1=<T<=49
    _,BW=cv2.threshold(H,t,255,cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(BW, cv2.MORPH_OPEN,kernel)
    return opening

def plot(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
