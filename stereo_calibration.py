import numpy as np
import cv2
import glob
import argparse
import sys
from param_storage import load_coefficients, save_stereo_coefficients

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
image_size = None

def load_image_points(dir, square_size, width=9, height=6):
    """Loads image points obtained from calibration images"""
    global image_size
    pattern_size = (width, height)

    # prepare the object points
    objp = np.zeros((height * width, 3), np.float32)
    objp[:,:2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)
    # create real world coordinates
    objp = objp * square_size

    # use arrays to store object and image points from all images
    objpoints = [] # 3D points in real world space
    left_imgpoints = [] # 2D points in image plane
    right_imgpoints = [] # 2D points in image plane

    # read images from directory
    left_images = glob.glob(dir + '/left_*.png')
    right_images = glob.glob(dir + '/right_*.png')

    # sort the images to make sure they're correctly ordered and paired
    left_images.sort()
    right_images.sort()

    # check for same number of images
    if len(left_images) != len(right_images):
        print('Images must be pairs. Number of right and left images not equal.')
        print('Left images count: ', len(right_images))
        print('Right images count: ', len(right_images))
        sys.exit(-1)
    
    paired_images = zip(left_images, right_images)

    for left_img, right_img in paired_images:
        # find right object points
        right = cv2.imread(right_img)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        # find left object points
        left = cv2.imread(left_img)
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

        if ret_left and ret_right:
            # object points
            objpoints.append(objp)
            # right points
            corners2_right = cv2.cornerSubPix(gray_right, corners_right, (5,5), (-1,-1), criteria)
            right_imgpoints.append(corners2_right)
            # left points
            corners2_left = cv2.cornerSubPix(gray_left, corners_left, (5,5), (-1,-1), criteria)
            left_imgpoints.append(corners2_left)
        else:
            print("Chessboard couldn't be detected. Image pair: " + left_img + " and " + right_img)
            continue

    image_size = gray_right.shape[::-1]

    return [objpoints, left_imgpoints, right_imgpoints]

def stereo_calibrate(left_file, right_file, dir, save_file, square_size, width=9, height=6):
    """Stereo calibration and rectification"""
    # load image points
    objp, leftp, rightp = load_image_points(dir, square_size, width, height)
    K1, D1, _t, _r = load_coefficients(left_file)
    K2, D2, _t, _r = load_coefficients(right_file)

    flag = 0
    flag |= cv2.CALIB_USE_INTRINSIC_GUESS

    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(objp, leftp, rightp, K1, D1, K2, D2, image_size)
    print("Stereo Calibration RMS: ", ret)
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

    save_stereo_coefficients(save_file, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera calibration')
    parser.add_argument('--left_file', type=str, required=True, help='left camera matrix file')
    parser.add_argument('--right_file', type=str, required=True, help='right camera matrix file')
    parser.add_argument('--dir', type=str, required=True, help='calibration images directory')
    parser.add_argument('--square_size', type=float, required=True, help='chessboard square length')
    parser.add_argument('--save_file', type=str, required=True, help='YML file to save stereo calibratin matrices')

    args = parser.parse_args()
    stereo_calibrate(args.left_file, args.right_file, args.dir, args.save_file, args.square_size)