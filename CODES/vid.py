# import the opencv library
import cv2
from preprocessing import *
from guide_lines import *

# define a video capture object
path1="/home/christian/Desktop/UNIV/DEMETER/VIDS/my_video-4.mp4"
vid = cv2.VideoCapture(path1)#,cv2.CAP_V4L2 )## 0 with rasp, 2 with pc
#vid.set(cv2.CAP_PROP_CONVERT_RGB, 0)##remove sustraction from camera
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
gridv=guide(frame_height,frame_width,angle=45,num=5)
gridh=guide(frame_height,frame_width,angle=0,num=7)


while(True):

    # Capture the video frame by frame
    ret, frame = vid.read()
    frame=cv2.addWeighted(frame, 1, gridv, 1, 0)
    #frame = cv2.addWeighted(frame, 1, gridh, 1, 0)
    #frame=greenbyHSV(frame)
    cv2.imshow("img",frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
