import cv2
import numpy as np

def guide(height, width ,angle=45,num=8,thickness=2,color=(255, 255, 0),img=None):
    if (img is not None):
        height, width, channels = img.shape
    hypot= int(1.1*np.sqrt(height**2 + width**2))
    hypot= (hypot+1) if (hypot%2 !=0) else hypot
    grid = np.zeros((hypot, hypot, 3), np.uint8)
    line_distance = int(hypot / (num + 1))

    for i in range(1, num + 1):
        y = i * line_distance
        cv2.line(grid, (0, y), (hypot, y), color, thickness)

    # get the center coordinates of the image to create the 2D rotation matrix
    cent = hypot // 2
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=(cent, cent), angle=angle, scale=1)
    # rotate the image using cv2.warpAffine
    rotated_image = cv2.warpAffine(src=grid, M=rotate_matrix, dsize=(hypot, hypot))

    mask=rotated_image[cent-height//2:cent+height//2,cent-width//2:cent+width//2]

    return mask

def plot_guide(img,mask):
    return cv2.addWeighted(img, 1, mask, 1, 0)
