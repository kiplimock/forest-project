# aligns left and right images
import cv2
import numpy as np
import matplotlib.pyplot as plt
from param_storage import load_stereo_coefficients

# load parameters
params = load_stereo_coefficients('stereo_params_v2.yml')
K1, D1, K2, D2, R, T = params[0:6]
R1, R2 = params[8:10]

# load images
# imgL = cv2.imread('images/stereo_data_set/setL.png', 0)
# imgR = cv2.imread('images/stereo_data_set/setR.png', 0)

imgL = cv2.imread('caps/left_11.png', 0)
imgR = cv2.imread('caps/right_11.png', 0)

size = imgL.shape[:2]

# # estimate rectification parameters
R1, R2, P1, P2, Q, validROI1, validROI2 = cv2.stereoRectify(K1, D1, K2, D2, size, R, T)

# rectification transformation maps
xmap1, ymap1 = cv2.initUndistortRectifyMap(K1, D1, R1, K1, size, cv2.CV_32FC1)
xmap2, ymap2 = cv2.initUndistortRectifyMap(K2, D2, R2, K2, size, cv2.CV_32FC1)

# now rectify the images
rectified_L = cv2.remap(imgL, xmap1, ymap1, cv2.INTER_LINEAR)
rectified_R = cv2.remap(imgR, xmap2, ymap2, cv2.INTER_LINEAR)

print('Rectified images saved successfully')

# grayLeft = cv2.cvtColor(rectified_L, cv2.COLOR_BGR2GRAY)
# grayRight = cv2.cvtColor(rectified_R, cv2.COLOR_BGR2GRAY)

# filter images
kernel_size = (7, 7)
filtered_L = cv2.GaussianBlur(imgL, kernel_size, 0)
filtered_R = cv2.GaussianBlur(imgR, kernel_size, 0)

# compute disparities
stereo1 = cv2.StereoBM_create(numDisparities=16, blockSize=5)
stereo2 = cv2.StereoBM_create(numDisparities=16, blockSize=7)
stereo3 = cv2.StereoBM_create(numDisparities=16, blockSize=13)
stereo4 = cv2.StereoBM_create(numDisparities=16, blockSize=15)

disparity1 = stereo1.compute(imgL, imgR)
disparity2 = stereo2.compute(imgL, imgR)
disparity3 = stereo3.compute(imgL, imgR)
disparity4 = stereo4.compute(imgL, imgR)
# disparity = stereo.compute(imgL, imgR)



# visualize results
# plt.figure(figsize=(10, 5))
# plt.subplot(231), plt.imshow(imgL, cmap='gray'), plt.title('Left Original'), plt.axis('off')
# # plt.subplot(232), plt.imshow(imgR, cmap='gray'), plt.title('Right Original'), plt.axis('off')
# plt.subplot(232), plt.imshow(rectified_L, cmap='gray'), plt.title('Left Rectified'), plt.axis('off')
# # plt.subplot(235), plt.imshow(rectified_R, cmap='gray'), plt.title('Right Rectified'), plt.axis('off')
# plt.subplot(233), plt.imshow(disparity1/2048, cmap='gray'), plt.title('Disparity, Blocksize=5'), plt.axis('off')
# plt.subplot(234), plt.imshow(disparity2/2048, cmap='gray'), plt.title('Disparity, Blocksize=7'), plt.axis('off')
# plt.subplot(235), plt.imshow(disparity3/2048, cmap='gray'), plt.title('Disparity, Blocksize=13'), plt.axis('off')
# plt.subplot(236), plt.imshow(disparity4/2048, cmap='gray'), plt.title('Disparity, Blocksize=15'), plt.axis('off')
# # plt.subplot(236), plt.imshow(filtered_L, cmap='gray'), plt.title('Left Filtered'), plt.axis('off')
# plt.tight_layout()
# plt.show()
