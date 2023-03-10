import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
import cv2

img=cv2.imread("./furrows.png")
r=img[:,:,0]#cv2.equalizeHist(img[:,:,0])
g=img[:,:,1]#cv2.equalizeHist(img[:,:,1])
b=img[:,:,2]#cv2.equalizeHist(img[:,:,2])
#img=cv2.merge((r,g,b))

T=r.copy()+g.copy()+b.copy()
T[T==0]=1

R=r.copy()/T 
G=g.copy()/T
B=b.copy()/T

print(R.dtype)

ExG = 2*G-R-B
ExR = 1.4*R-G
ExGR = ExG-ExR 
CIVE = 0.441*R-0.811*G + 0.385*G + 18.78745
VEG = G/(R**0.667 * B**0.333)

COM = 0.25*ExG+ 0.30*ExGR+ 0.33*CIVE + 0.12*VEG*R

cv2.imshow("com",COM)
cv2.waitKey(0)
cv2.destroyAllWindows()