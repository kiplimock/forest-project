 # this script performs the calibration for a   single camera
import numpy as np
import cv2
import glob
import argparse
from param_storage import load_coefficients, save_coefficients, save_points, load_points
from projection_error import projection_error as error

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(path, square_size, width=9, height=6):
    """perform calibration using images in the given path"""

    # prepare object points, like (0,0,0), (2,0,0) etc
    objp = np.zeros((height*width, 3), np.float32)
    objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

    objp = objp * square_size # create real world coordinates

    # store object points and image points from all images in arrays
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane

    # loop through all the images
    # discard image if opencv cannot find corners
    images = glob.glob(path + '/' + "*.png")

    for fname in images:
        # read all images
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # if corners are found, add object and image points after refining them
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # draw and display the chessboard corners
            cv2.drawChessboardCorners(gray, (width, height), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[0], img.shape[1]), None, None)
        save_points('OP', objpoints[0], 'right_points.yml')
        save_points('IP', imgpoints[0], 'right_points.yml')

        return [ret, mtx, dist, rvecs, tvecs]

if __name__ == "__main__":
    # check the help parameters to understand arguments
    # parser = argparse.ArgumentParser(description='Camera calibration')
    # parser.add_argument('--img_dir', type=str, required=True, help='image directory path')
    # parser.add_argument('--square_size', type=float, required=False, help='chessboard square size')
    # parser.add_argument('--width', type=int, required=False, help='chessboard width size, default is 9')
    # parser.add_argument('--height', type=int, required=False, help='chessboard height size, default is 6')
    # parser.add_argument('--save_file', type=str, required=True, help='YML file to save calibration matrices')

    # args = parser.parse_args()

    # call the calibration function and save as file
    ret, mtx, dist, rvecs, tvecs = calibrate(path='stereoCalibrationImages/right', square_size=2.3)
    object_points = load_points('OP', 'right_points.yml')
    image_points = load_points('IP', 'right_points.yml')

    save_coefficients(mtx, dist, tvecs[0], rvecs[0], 'right_cam.yml')
    parameters = load_coefficients('right_cam.yml')
    p_error = error(object_points, image_points,  parameters[2],  parameters[3],  parameters[0],  parameters[1])

    print("Calibration is finished. RMS: ", ret)
    print("Mean Projection Error: ", p_error)