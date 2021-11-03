import numpy as np
from matplotlib import pyplot as plt
import cv2

imgL = cv2.imread('stereoCalibrationImages/left_0.jpg', 0)
imgR = cv2.imread('stereoCalibrationImages/right_0.jpg', 0)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(imgL, imgR)
plt.imshow(disparity, 'gray')
plt.show()