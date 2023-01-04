import cv2
import numpy as np
import matplotlib.pyplot as plt
from code.constants import *

def PnPransacCV(P1, X1, K):
    _, rvec, tvec, inliers = cv2.solvePnPRansac(X1, P1, K, distCoeffs=None)
    RotMat = cv2.Rodrigues(rvec)[0]

    return RotMat, tvec

def PnPRANSAC(P1, X1, K):
    '''
    Inputs
        - P1 = Keypoints as homogenous coordinates in the image plane
        - X1 = 3D Points corresponding to the keypoints
        - K  = Camera calibration matrix
    Outputs
        - R1 = Rotation from world to the camera frame (maybe the other way around?)
        - T1 = Translation from the world to the camera frame
    '''
    
    N_inliers = 0
    
    for i in range(NUM_ITER_RANSAC):
        # TODO: Check the dimensions
        idx_RANSAC = np.random.choice(P1.shape[1], 4, replace=False)        
        # Using PnP from OpenCV (https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
        print(P1[:,idx_RANSAC].T.shape)
        _, R1, T1 = cv2.solvePnP(objectPoints=X1[:,idx_RANSAC].T,
                                 imagePoints=P1[:,idx_RANSAC].T,
                                 cameraMatrix=K,
                                 distCoeffs=None)
        
        # Project 3D points into the image plane using the estimated R and T
        X1_proj = projectPoints(X1, K)
        # Calulate inliers
        dist = X1_proj.T - P1
        errors = (dist**2).sum(0)
        
        idx_inlier = errors < PIXEL_TOLERANCE**2
        n_in = np.count_nonzero(idx_inlier)
        if n_in > N_inliers:
            N_inliers = n_in
            _, R1_best, T1_best = cv2.solvePnP(objectPoints=X1[:,idx_inlier],
                                               imagePoints=P1[:,idx_inlier],
                                               cameraMatrix=K,
                                               distCoeffs=None)
    
    return R1_best, T1_best


def projectPoints(points_3d, K, D=np.zeros([4, 1])):
    """
    Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
    distortion coefficients (4x1).
    """
    # get image coordinates
    projected_points = np.matmul(K, points_3d[:, :, None]).squeeze(-1)
    projected_points /= projected_points[:, 2, None]

    # apply distortion
    # # projected_points = distortPoints(projected_points[:, :2], D, K)

    return projected_points