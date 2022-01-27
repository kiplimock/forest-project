import cv2
import numpy as np
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class SceneReconstruction3D:
    def __init__(self, K, dist):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.d = dist

    def load_image_pair(self, img_path1, img_path2, downscale=True):
        self.img1 = cv2.imread(img_path1, cv2.CV_8UC3) # read image as 3 channel BGR
        self.img2 = cv2.imread(img_path2, cv2.CV_8UC3)

        # validate the images
        if self.img1 is None:
            sys.exit("Image " + img_path1 + " could not be loaded.")
        if self.img2 is None:
            sys.exit("Image " + img_path2 + " could not be loaded.")

        # convert grayscale images to BGR format
        if len(self.img1.shape) == 2:
            self.img1 = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
            self.img2 = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)

        # downscale larger images to about 600px wide
        target_width = 600
        if downscale and self.img1.shape[1] > target_width:
            while self.img1.shape[1] > 2*target_width:
                self.img1 = cv2.pyrDown(self.img1)
                self.img2 = cv2.pyrDown(self.img2)

        # undo any radial and tangential distortions
        self.img1 = cv2.undistort(self.img1, self.K, self.d)
        self.img2 = cv2.undistort(self.img2, self.K, self.d)

    def __extract_keypoints(self, feat_mode):
        if feat_mode.lower == "surf":
            # feature matching using SURF and BFMatcher
            self._extract_keypoints_surf()
        else:
            if feat_mode.lower() == "flow":
                # feature matching using optic flow
                self._extract_keypoints_flow()
            else:
                sys.exit("Unknown mode " + feat_mode + ". Use 'SURF' or 'FLOW'")
    
    # point matching using SURF
    def _extract_keypoints_surf(self):
        '''Matches keypoints using SURF descriptors'''
        detector = cv2.xfeatures2d.SURF_create(250)
        first_keypoints, first_des = detector.detectAndCompute(self.img1, None)
        second_keypoints, second_des = detector.detectAndCompute(self.img2, None)

        # feature matching using Brute Force
        # FLANN can also work here
        matcher = cv2.BFMatcher(cv2.NORM_L1, True)
        matches = matcher.match(first_des, second_des)

        # for each match we recover corresponding image coordinates
        first_match_points = np.zeros((len(matches), 2), dtype=np.float32)
        second_match_points = np.zeros_like(first_match_points)

        for i in range(len(matches)):
            first_match_points[i] = first_keypoints[matches[i].queryIdx].pt
            second_match_points[i] = second_keypoints[matches[i].trainIdx].pt

        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

    # point matching using optical flow
    def _extract_keypoints_flow(self):
        '''Matches keypoints by tracking their displacement'''
        # get the FAST features
        fast = cv2.FastFeatureDetector_create()
        first_keypoints = fast.detect(self.img1, None)

        # compute the optical flow of the features
        # the output will be a list of corresponding features in second image
        first_key_list = [i.pt for i in first_keypoints]
        first_key_arr = np.array(first_key_list).astype(np.float32)
        second_key_arr, status, err = cv2.calcOpticalFlowPyrLK(self.img1, 
        													   self.img2, 
        													   first_key_arr, 
        													   None)

        # exclude points whose status is 0 or error is above a threshold
        condition = (status == 1) * (err < 5.)
        concat = np.concatenate((condition, condition), axis=1)
        first_match_points = first_key_arr[concat].reshape(-1,2)
        second_match_points = second_key_arr[concat].reshape(-1,2)

        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

    # draw the optical flow field
    def plot_optic_flow(self):

        self.__extract_keypoints("flow")

        img = self.img1
        for i in range(len(self.match_pts1)):
            cv2.line(img, tuple(self.match_pts1[i]), tuple(self.match_pts2[i]), 
            		 color=(255,0,0))

            theta = np.arctan2(self.match_pts2[i][1] - self.match_pts1[i][1],
            				   self.match_pts2[i][0] - self.match_pts1[i][0])

            cv2.line(img, tuple(self.match_pts2[i]),
            		(np.int(self.match_pts2[i][0] - 6*np.cos(theta + np.pi/4)), 
            		 np.int(self.match_pts2[i][1] - 6*np.sin(theta + np.pi/4))),
            		 color=(255,0,0))

            cv2.line(img, tuple(self.match_pts2[i]),
            		(np.int(self.match_pts2[i][0] - 6*np.cos(theta - np.pi/4)),
            		 np.int(self.match_pts2[i][1] - 6*np.sin(theta - np.pi/4))),
            		 color=(255,0,0))

        cv2.imshow("imgFlow", img)
        cv2.waitKey()

    # fundamental matrix F
    def _find_fundamental_matrix(self):
        self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2, 
                                                    cv2.FM_RANSAC, 0.1, 0.99)

    # essential matrix E
    def _find_essential_matrix(self):
        self.E = self.K.T.dot(self.F).dot(self.K)

    # decompose essential matrix into rotational and translational components [R|t]
    def _find_camera_matrices(self):
        """
        Computes the [R|t] camera matrix
        """
        U, S, Vt = np.linalg.svd(self.E) # U & V are unitary matrices
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        # U, V & W together are used to reconstruct [R|t]

        # iterate over all the point correspondences used in estimating
        # the fundamental matrix
        first_inliers = []
        second_inliers = []
        for i in range(len(self.Fmask)):
            if self.Fmask[i]:
                # convert keypoints from 2D to homogeneous coordinates
                first_inliers.append(self.K_inv.dot([self.match_pts1[i][0],
                                     self.match_pts1[i][1], 1.0]))
                second_inliers.append(self.K_inv.dot([self.match_pts2[i][0],
                                      self.match_pts2[i][1], 1.0]))
        
        # determine the correct choice of theh camera matrix
        # only in one of the 4 possible configurations will all the points
        # be in front of both cameras
        # First Choice: R = U * Wt * Vt, T = +u_3 (Hartley & Zisserman 9.19)
        R = U.dot(W).dot(Vt)
        T = U[:, 2]
        if not self._in_front_of_both_cameras(first_inliers, second_inliers, R, T):
            # Second Choice: R = U * W * Vt, T = -u_3
            T = - U[:, 2]
        
        if not self._in_front_of_both_cameras(first_inliers, second_inliers, R, T):
            # Third Choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]

            if not self._in_front_of_both_cameras(first_inliers, second_inliers, R, T):
                # Fourth Choice: R = U * Wt * Vt, T = -u_3
                T = - U[:, 2]

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    # validate keypoint pairs by making sure they lie in front of both cameras
    def _in_front_of_both_cameras(self, first_points, second_points, rot, trans):
        """
        Determines whether point correspondences are in front of both cameras
        """
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :], 
                             trans) / np.dot(rot[0, :] - second[0] * rot[2, :], 
                             second)
            first_3d_point = np.array([first[0] * first_z, 
                                      second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False
            return True
        
    # perform image rectification
    def _plot_rectified_images(self, feat_mode="SURF"):
        """
        Plots Rectified images
        This method computes and plots a rectified version of the two images side by side
        """
        self.__extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices()

        R = self.Rt2[:, :3]
        T = self.Rt2[:, 3]

        # perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(self.K, self.d,
                                                          self.K, self.d, 
                                                          self.img1.shape[:2],
                                                          R, T, alpha=1.0)

        mapx1, mapy1 = cv2.initUndistortRectifyMap(self.K, self.d, R1, self.K,
                                                   self.img1.shape[:2], 
                                                   cv2.CV_32F)
        mapx2, mapy2 = cv2.initUndistortRectifyMap(self.K, self.d, R2, self.K,
                                                   self.img2.shape[:2], 
                                                   cv2.CV_32F)

        img_rect1 = cv2.remap(self.img1, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect2 = cv2.remap(self.img2, mapx2, mapy2, cv2.INTER_LINEAR)

        # plot the two images next to each other
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

        # draw horizontal blue lines every 25px across the side-by-side image
        for i in range(20, img.shape[0], 25):
            cv2.line(img, (0, i), (img.shape[1], i), (255,0,0))

        plt.subplot(121), plt.imshow(self.img1), plt.axis('off')
        plt.subplot(122), plt.imshow(self.img2), plt.axis('off')
        plt.show()

        # cv2.imshow('imgRectified', img)
        # cv2.waitKey()


