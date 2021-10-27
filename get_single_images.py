import numpy as np
import cv2
import time
import sys

# set the camera
cap = cv2.VideoCapture(0)

i = 0

def main():
    global i
    if len(sys.argv) < 4:
        print("Usage: ./program_name directory_to_save start_index prefix")
        sys.exit(1)
    
    i = int(sys.argv[2])
    while True:
        # capture frame by frame
        ret, frame = cap.read()

        # display the frame
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite(sys.argv[1] + '/' + sys.argv[3] + str(i) + ".png", frame)
            i += 1
    
    # now release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()