import cv2 
import numpy as np 
import glob

pixl=0
click=True

def mouse(event, x, y, flags, params):
    global pixl,click
    if event == cv2.EVENT_MOUSEMOVE:
        pixl=(x,y)
    #if event == cv2.EVENT_LBUTTONUP:
    #    click= False

res=5
step=2.5

#text
font = cv2.FONT_HERSHEY_SIMPLEX
org = (10, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 1

path1="/home/christian/Desktop/UNIV/DEMETER/VIDS/my_video-4.mp4"
path2="/home/christian/Desktop/UNIV/DEMETER/TRAINING/PLANT-SOIL/"

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

pixl=(width//2,height//2)
while(vid_capture.isOpened()):
    # and the second is frame
    ret, frame = vid_capture.read()
    if (ret==True):
        cv2.imshow("window",frame)
        cv2.waitKey(1)
        while(click):
            cv2.setMouseCallback("window", mouse)
            frame2=cv2.line(frame.copy(),(width//2,height),pixl,(0,0,250),5)

            try:
                angle = np.arctan((height-pixl[1] )/(pixl[0]- width/2)) * 180 / np.pi
            except:
                angle=90

            angle = (angle + 180) if (angle < 0) else angle
            angle= int ((angle//res)*res + ((angle%res)//step)*res)

            angle = 45 if (angle < 45) else angle
            angle = 135 if (angle > 135) else angle

            frame2 = cv2.putText(frame2, str(angle), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow("window", frame2)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("o"):
                break
            if key == ord("q"):
                exit()
            elif key == ord("s"):
                count = len(glob.glob1(path2 + str(angle) + "/", '*.png'))
                cv2.imwrite("%s/%d/%d.png" % (path2, angle, count), frame)
                break
        click=True
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()

