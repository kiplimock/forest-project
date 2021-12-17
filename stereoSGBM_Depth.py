import numpy as np
import cv2 as cv


def main():
    print('loading images...')
    imgL = cv.pyrDown(cv.imread(cv.samples.findFile('stereoCalibrationImages/outdoors/left_4.jpg')))
    imgR = cv.pyrDown(cv.imread(cv.samples.findFile('stereoCalibrationImages/outdoors/left_5.jpg')))

    # disparity range is tuned for image pair
    window_size = 3
    min_disp = -1
    num_disp = 15 - min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 7,
        speckleWindowSize = 50,
        speckleRange = 2
    )

    # stereo2 = cv.StereoBM_create(numDisparities=16, blockSize=5)

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp - min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()