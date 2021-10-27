import numpy as np
import cv2
import glob
from param_storage import load_coefficients

# load camera matrix
K1, D1, _t, _r = load_coefficients('single_cam.yml')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
# X, Y & Z axes
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

def draw_axis(img, corners, imgpts):
    """Draws a 3D Coordinate axis at Chessboard Corner"""
    pt1 = (int(corners[0][0][0]), int(corners[0][0][1]))

    # you encounter a parsing error when you pass the image points as an array
    # of float values. Hence the conversion to int
    pt2 = [(int(imgpts[0][0][0]), int(imgpts[0][0][1])), (int(imgpts[1][0][0]), int(imgpts[1][0][1])), (int(imgpts[2][0][0]), int(imgpts[2][0][1]))]
    print(type(pt1) == type(pt2[0]))
    img = cv2.line(img, pt1, pt2[0], (255, 0, 0), 5)
    img = cv2.line(img, pt1, pt2[1], (0, 255, 0), 5)
    img = cv2.line(img, pt1, pt2[2], (0, 0, 255), 5)
    return img

# we draw the axes in each image
i = 0
for fname in glob.glob('stereoCalibrationImages/left_1*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # find rotation and translation matrices
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, K1, D1)

        #project 3D points to the image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, K1, D1)
        img = draw_axis(img, corners2, imgpts)
        cv2.imshow('img', img)
        key =cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            cv2.imwrite('pose/pose_' + str(i) + '.png', img)
            i += 1

cv2.destroyAllWindows()