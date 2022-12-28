import cv2
import numpy as np
import matplotlib.pyplot as plt

from track_klt_robustly import trackKLTRobustly

def KLT(keypoints, I, I_prev) -> np.ndarray:
    """ 
    Input:
        keypoints   np.ndarray, n x 2, n points to track as [x y] = [col row]
        I_prev      previous image
        I           current image
    Constants:
        r_T         scalar, radius of patch to track
        n_iter      scalar, number of iterations
        threshold   scalar, bidirectional error threshold
    Output:
        keypoints         np.ndarray, m x 2, of m new points
    """ 
    r_T = 15
    n_iter = 50
    threshold = 0.1

    dkp = np.zeros_like(keypoints)
    # dkp = []
    keep = np.ones((keypoints.shape[1],)).astype('bool')
    for j in range(keypoints.shape[1]):
        kptd, k = trackKLTRobustly(I_prev, I, keypoints[:,j].T, r_T, n_iter, threshold)
        if k:
            dkp.append(kptd)
        dkp[:, j] = kptd
        keep[j] = k
    # kpold = keypoints[:, keep]
    keypoints = keypoints + dkp
    keypoints = keypoints[:, keep]

    return keypoints