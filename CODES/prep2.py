import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import disk
import os

img=cv2.imread("/home/christian/Desktop/UNIV/DEMETER/CODES/12.png",cv2.IMREAD_GRAYSCALE)

#r=(img[:,:,0])
#g=(img[:,:,1])
#b=(img[:,:,2])

height, width = img.shape[:2]
roi_height = height // 3

# Create a figure to plot the vertical projections
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# Get the vertical projection of each section
for i in range(3):
    # Define the region of interest (ROI)
    y1 = i * roi_height
    y2 = (i + 1) * roi_height
    roi = img[y1:y2, :]

    # Get the vertical projection of the ROI
    vertical_projection = np.sum(roi, axis=0)

    # Plot the vertical projection
    axs[i].plot(vertical_projection)
    axs[i].set_xlabel('Column')
    axs[i].set_ylabel('Sum of Pixel Values')
    axs[i].set_title('Vertical Projection - Region {}'.format(i + 1))

# Adjust the layout of the plots
plt.tight_layout()

# Show the figure
plt.show()

""" 
# Path to the folder containing the images
folder_path = '/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-CONCRETE-PREP/90/'

# Get the list of image file names in the folder
file_names = sorted(os.listdir(folder_path))

# Initialize the Matplotlib figure
fig, ax = plt.subplots()

# Loop over the images
for file_name in file_names:
    # Load the image
    img = cv2.imread(os.path.join(folder_path, file_name),cv2.IMREAD_GRAYSCALE)
    img = cv2.distanceTransform(255 - img.copy(), cv2.DIST_L2, 3).astype(np.uint8)
    _,img_ = cv2.threshold(img.copy(),50,255,cv2.THRESH_BINARY)

    # Plot the image
    ax.imshow(img)
    ax.set_title(file_name)

    # Pause for a short time to simulate video playback
    plt.pause(0.01)

    # Clear the figure for the next frame
    ax.clear()

# Close the figure
plt.close()

edges = cv2.Canny(img, 50, 150, apertureSize=3)

# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=200,  # Min number of votes for valid line
    minLineLength=50,  # Min allowed length of line
    maxLineGap=50  # Max allowed gap between line for joining them
)

# Iterate over points
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines joing the points
    # On the original image
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

# Save the result image
cv2.imshow('detectedLines.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 

"""