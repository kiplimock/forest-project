import cv2
from param_storage import save_points, load_points

def projection_error(objpoints, imgpoints, tvecs, rvecs, mtx, dist):
    mean_error = 0

    # generate 2D points from 3D points and camera parameters
    imgpoints2, _ = cv2.projectPoints(objpoints, rvecs, tvecs, mtx, dist)
    save_points('IP2', imgpoints2, 'points.yml')
    # compute error between projected points and original points
    error = cv2.norm(imgpoints, imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
    
    return mean_error/len(objpoints)