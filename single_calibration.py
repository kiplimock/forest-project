 # this script performs the calibration for a   single camera
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from param_storage import  save_coefficients
from turtle import color

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate(path, square_size, width=9, height=6):
    """
    This function performs calibration using calibration images
    in the given path
    """

    # prepare object points, like (0,0,0), (2,0,0) etc
    objp = np.zeros((height*width, 3), np.float32)
    objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)

    objp = objp * square_size # create real world coordinates

    # store object points and image points from all images in arrays
    objpoints = [] #3D points in real world space
    imgpoints = [] #2D points in image plane

    # loop through all the images
    # discard image if opencv cannot find corners
    global images
    images = glob.glob(path + '/' + "*.jpg")

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

    flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO +        cv2.CALIB_ZERO_TANGENT_DIST 

    mtx_init = np.array([[1500, 0, 640], [0, 1500, 360], [0, 0, 1]])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                       imgpoints, 
                                                       (1280, 720),
                                                       mtx_init,
                                                       None,
                                                       flags=flags)

    return [ret, mtx, dist, rvecs, tvecs, imgpoints, objpoints]


def projection_error(objpoints, imgpoints, tvecs, rvecs, mtx, dist):
    """
    This function computes the backprojection error in order
    to estimate the accuracy of the parameters found during
    calibration
    """
    mean_error = 0
    image_errors = []
    # generate 2D points from 3D points and camera parameters
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

        # compute error between projected points and original points
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        image_errors.append(error)
        mean_error += error
    
    ME = mean_error/len(objpoints)
    return {'rms' :ME, 'Image errors': image_errors}

def plot_scatter(objpoints, imgpoints, tvecs, rvecs, mtx, dist):
    """
    This function creates a scatterplot of the residual errors
    due to differences between original and reprojected image points
    """
    # initialize list of errors
    errors = []
    x = [] # list of x ordinates
    y = [] # list of y ordinates


    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) 
        # --> len(imgpoints2) = 54
        error = [np.subtract(imgpoints2[j][0], imgpoints[i][j][0]) for j in range(len(imgpoints2))] # --> [x,y]
        errors.append(error) # --> [[x1,y1], [x2,y2], ..., [xn,yn]]
    
        x.append([pair[0] for pair in error])
        y.append([pair[1] for pair in error])
        
    # plot figure
    plt.figure(figsize=(10, 6))
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Reprojection Errors in Pixels")

    labels = ["Image " + str(i) for i in range(len(images))]
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(len(imgpoints))]
    for i in range(len(colors)):
        # different colour for each image
        # labels and colours correspond
        plt.scatter(x[i], y[i], marker='+', color=colors[i], label=[labels[i]])
    
    # plot legend outside the image
    plt.legend(bbox_to_anchor=(1.05,1.0), loc='upper left')
    plt.tight_layout()
    plt.text(-1.4, 1.0, 'RMSE: ' + str(round(ret, 4)))
    plt.show()




if __name__ == "__main__":

    ret, mtx, dist, rvecs, tvecs, image_points, object_points = calibrate(path='images/cam2/take3', square_size=2.3)

    save_coefficients(mtx, dist, tvecs[0], rvecs[0], 'cam.yml')
    p_error = projection_error(object_points, image_points, tvecs, rvecs, mtx, dist)

    print("Calibration is finished. RMS: ", ret)
    print("Error Information: ", p_error)
    plot_scatter(object_points, image_points, tvecs, rvecs, mtx, dist)