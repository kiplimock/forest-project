# this code draws extreme lines on a pair of images
import numpy as np
import cv2
from matplotlib import pyplot as plt

# read image pair
img1 = cv2.imread('stereoCalibrationImages/left_2.jpg')
img2 = cv2.imread('stereoCalibrationImages/right_2.jpg')

# find the key points and descriptors using SIFT
sift = cv2.SIFT()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# compute FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []

# ratio test
for i, (m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(n)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.trainIdx].pt)
        