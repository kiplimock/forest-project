# captures images of the left and right cameras

import numpy as np
import cv2
import argparse
import glob
import sys

# set values for cameras
capL = cv2.VideoCapture(1)
capR = cv2.VideoCapture(0)

i = 0 # image index

def main():
    global i
    if len(sys.argv) < 3:
        print("Usage: ./program_name directory_to_save start_index")
        sys.exit()
    
    i = int(sys.argv[2]) # get the start index

    while True:
        # Grab and retrieve for sync
        if not (capL.grab() and capR.grab()):
            print("No more frames")
            break

        _, leftFrame = capL.retrieve()
        _, rightFrame = capR.retrieve()

        cv2.imshow('capL', leftFrame)
        cv2.imshow('capR', rightFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite(sys.argv[1] + "/left/left_" + str(i) + ".jpg", leftFrame)
            cv2.imwrite(sys.argv[1] + "/right/right_" + str(i) + ".jpg", rightFrame)
            i += 1
    
    capL.release()
    capR.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()