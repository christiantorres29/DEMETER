# Python program to save a
# video using OpenCV
import cv2
import numpy as np
import glob
import tqdm
from guide_lines import *


angle=int(input("Angulo deseado de guía: "))
path1="/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-CONCRETE/"
#count = len(glob.glob1(path1, str(angle) + '-*.mp4'))

#text
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 50)
fontScale = 1
color = (70, 0, 255)
thickness = 3

# Create an object to read
# from camera
video = cv2.VideoCapture(2)

# We need to check if camera
# is opened previously or not
if (video.isOpened() == False):
	print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))
grid=guide(frame_height,frame_width,angle=angle,num=3)
grid90=guide(frame_height,frame_width,angle=90,num=3)

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
#result = cv2.VideoWriter(path1+str(angle)+"-"+str(count)+str(".mp4"),
#						cv2.VideoWriter_fourcc(*'mp4v'),
#						60, size)

frames=0

frame_count=input("Frames por video: ")
frame_count=1000 if frame_count=='' else frame_count
frame_count=int(frame_count)

while(True):
	ret, frame = video.read()
	frame = cv2.putText(frame, "TEST", org, font,
				fontScale, color, thickness, cv2.LINE_AA)
	if ret == True:
		frame2 = cv2.addWeighted(frame.copy(), 1, grid, 1, 0)
		frame2 = cv2.addWeighted(frame2, 1, grid90, 1, 0)
		cv2.imshow('Frame', plot_guide(frame2,grid))
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			exit()
		elif key==ord("s"):
			break
	else:
		break


while(frames<frame_count):
	ret, frame = video.read()
	if ret == True:
		#result.write(frame)
		cv2.imshow('Frame', plot_guide(frame,grid))
		frames+=1
		print(frames)
		count = len(glob.glob1(path1 + str(angle) + "/", '*.png'))
		cv2.imwrite("%s/%d/%d.png" % (path1, angle, count), frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			exit()

	else:
		break

video.release()
#result.release()

# Closes all the frames
cv2.destroyAllWindows()



print("The video was successfully saved")
