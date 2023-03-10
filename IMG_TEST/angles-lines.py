import cv2 
import numpy as np 
import glob

pixl=0
def mouse(event, x, y, flags, params):
    global pixl
    if event == cv2.EVENT_MOUSEMOVE:
        pixl=(x,y)

res=5

path1="/home/christian/Desktop/UNIV/DEMETER/VIDS/my_video-5.mp4"
path2="/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-CONCRETE/video5/"

# Create a video capture object, in this case we are reading the video from a file

vid_capture = cv2.VideoCapture(path1)
    
if (vid_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    print('Frames per second : ', fps,'FPS')
    
    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    height=int (vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width =int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("height : "+str(height))
    print("width : "+str(width))
#---------------------------------------------------------------------------------------

count= len(glob.glob1(path2,'frame*.png'))

pixl=(width//2,height//2)
while(vid_capture.isOpened()):
    # and the second is frame
    ret, frame = vid_capture.read()
    if (ret==True):
        cv2.imshow("window",frame)
        cv2.setMouseCallback("window", mouse)
        frame2=cv2.line(frame.copy(),(width//2,height),pixl,(0,0,250),5)
        cv2.imshow("window",frame2)

        angle=np.arctan( (pixl[0]- width/2) / (height-pixl[1] ))*180/np.pi

        angle= int ((angle//res)*res)
        print(angle)

        cv2.imwrite("%s/frame%d_%d.png" %(path2,count,angle), frame)
        count+=1
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()

