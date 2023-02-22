import cv2 
import argparse
import time
 
def vid2frame(path1, path2):
    # Create a video capture object, in this case we are reading the video from a file

    print(path1)
    print(path2)

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
        print('Frame count : ', frame_count)
     
    count=0
    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool 
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imwrite("%s/frame%d.png" %(path2,count), frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe

        if(count==(frame_count-1)):
            break

        count+=1

    # Release the video capture object
    vid_capture.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path1', help='Path of input file')
    parser.add_argument('path2', help='Path to output files')
    args = parser.parse_args()
    vid2frame(args.path1,args.path2) 
