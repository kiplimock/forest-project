# this file stores functions for saving and retrieving 
# camera and stereo coefficients

import cv2

def save_coefficients(mtx, dist, tvec, rvec, path):
    """
    Saves the camera matrix and distortion coefficients to a given path/file
    """
    coeff_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    coeff_file.write("K", mtx)
    coeff_file.write("D", dist)
    coeff_file.write("T", tvec)
    coeff_file.write("R", rvec)
    
    coeff_file.release()

def save_points(id, points, path):
    """
    Saves the object and/or image points to a given path/file
    """
    coeff_file = cv2.FileStorage(path, cv2.FILE_STORAGE_APPEND)
    coeff_file.write(id, points)
    
    coeff_file.release()

def save_stereo_coefficients(path, K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q):
    """Save the coefficcients of the stereo system to file"""
    file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    file.write("K1", K1)
    file.write("D1", D1)
    file.write("K2", K2)
    file.write("D2", D2)
    file.write("R", R)
    file.write("T", T)
    file.write("E", E)
    file.write("F", F)
    file.write("R1", R1)
    file.write("R2", R2)
    file.write("P1", P1)
    file.write("P2", P2)
    file.write("Q", Q)
    file.release()

def load_coefficients(path):
    """Loads camera matrix and distortion coefficients from file"""
    coeff_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    # retrieve the FileNode object as a matrix
    mtx = coeff_file.getNode("K").mat()
    dist = coeff_file.getNode("D").mat()
    tvec = coeff_file.getNode("T").mat()
    rvec = coeff_file.getNode("R").mat()

    coeff_file.release()
    return [mtx, dist, tvec, rvec]

def load_stereo_coefficients(path):
    """Loads stereo coefficients from file"""
    # read file from storage
    file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    # we load them as matrices
    K1 = file.getNode("K1").mat()
    D1 = file.getNode("D1").mat()
    K2 = file.getNode("K2").mat()
    D2 = file.getNode("D2").mat()
    R = file.getNode("R").mat()
    T = file.getNode("T").mat()
    E = file.getNode("E").mat()
    F = file.getNode("F").mat()
    R1 = file.getNode("R1").mat()
    R2 = file.getNode("R2").mat()
    P1 = file.getNode("P2").mat()
    P2 = file.getNode("P2").mat()
    Q = file.getNode("Q").mat()

    # release the file
    file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]

def load_points(id, path):
    """Load object and/or image points from file"""
    file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    points = file.getNode(id).mat()

    file.release()
    return points