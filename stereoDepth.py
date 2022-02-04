import numpy as np
import matplotlib.pyplot as plt
import cv2

# read both images and convert to grayscale
imgL = cv2.imread("caps/left_11.png", 0)
imgR = cv2.imread("caps/right_11.png", 0)

# --------------------------------------------- #
# PREPROCESSING
# --------------------------------------------- #

# Compare both preprocessed images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(imgL)
axes[1].imshow(imgR)
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
plt.suptitle("Original Images")
plt.show()

# 1.Detect keypoints and their descriptors

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(imgL, None)
kp2, des2 = sift.detectAndCompute(imgR, None)

# plot the keypoints
imgSift = cv2.drawKeypoints(imgL, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("SIFT Keypoints", imgSift)

# Match the keypoints in both images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Apply the D.G Lowe ratio test to keep good matches
matchesMask = [[0, 0] for i in range(len(matches))]
good = []
pts1 = []
pts2 = []

for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        # keep this keypoint pair
        matchesMask[i] = [1, 0]
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


# Draw the keypoiny matches between both pictures
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask[300:500],
                   flags=cv2.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv2.drawMatchesKnn(imgL, kp1, imgR, kp2, matches[300:500], None, **draw_params)
cv2.imshow("Keypoint Matches", keypoint_matches)

# ------------------------------------- #
# STEREO RECTIFICATION
# ------------------------------------- #

# Calculate the fundamental matrix for the cameras
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# We take only the inlier points
pts1 = pts1[inliers.ravel() == 1]
pts2 = pts2[inliers.ravel() == 1]

# Visualize epilines
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    '''
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    '''
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)

    # Edit: use the same random seed so that the two images are comparable
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


# Find epilines corresponding to points in the right image (second image) and 
# drawing its lines on the left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(imgL, imgR, lines1, pts1, pts2)

# Find epilines corresponding to points in the right image (second image) and 
# drawing its lines on the left image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 2, fundamental_matrix)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(imgR, imgL, lines2, pts2, pts1)

plt.subplot(121), plt.imshow(img5)
plt.subplot(122), plt.imshow(img3)
plt.suptitle("Epilines in both images")
plt.show()

# Stereo Rectification
h1, w1 = imgL.shape
h2, w2 = imgR.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1,h1))

# Rectify the Images
imgL_rectified = cv2.warpPerspective(imgL, H1, (w1, h1))
imgR_rectified = cv2.warpPerspective(imgR, H2, (w2, h2))

# Plot the Rectified Images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(imgL_rectified, cmap='gray')
axes[1].imshow(imgR_rectified, cmap='gray')
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
plt.suptitle("Rectified Images")
plt.show()

# -------------------------------- #
# COMPUTE DISPARITY MAP
# -------------------------------- #

# Matched blocked size
block_size = 11 # must be odd and > 1. Normally in the range [3, 11]
min_disp = -128
max_disp = 128

# max_disp - min_disp must always > 0 and be divisible by 16
num_disp = max_disp - min_disp

# % margin by which minimum cost function beast the second best to be considered a correct match
# range [5, 15] is good
uniquenessRatio = 5 

# max size of smooth disparity regions to consider their noise speckles and invalidate
# range [50, 200] is good. 0 disables speckle filtering
speckleWindowSize = 200

# maximum disparity variation within each connected component. 1 or 2 is good
speckleRange = 2

# Maximum allowed difference (in integer pixel units) in the left-right disparity check
# 0 disables the check
disp12MaxDiff = 0

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * block_size * block_size,
    P2=32 * block_size * block_size
)

disparity_SGBM = stereo.compute(imgL_rectified, imgR_rectified)

plt.imshow(disparity_SGBM, cmap='plasma')
plt.colorbar()
plt.show()

# Normalize the values to the range 0-255 for a grayscale image
disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
plt.imshow(disparity_SGBM, cmap='gray')
plt.show()