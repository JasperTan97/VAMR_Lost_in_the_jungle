import cv2
import numpy as np
import matplotlib.pyplot as plt

from code.track_klt_robustly import trackKLTRobustly
from code.constants import *

def KLT_bootstraping(kpt_original, keypoints, I, I_prev):
    """ 
    Input:
        kpt_original        np.ndarray, n x 2, from the first image
        keypoints           np.ndarray, n x 2, n points to track as [x y] = [col row]
        I_prev              previous image
        I                   current image
    Constants:
        r_T                 scalar, radius of patch to track
        n_iter              scalar, number of iterations
        threshold           scalar, bidirectional error threshold
    Output:
        keypoints           np.ndarray, m x 2, of m new points
    """ 

    keypoints = keypoints.T
    kpt_original = kpt_original.T
    dkp = np.zeros_like(keypoints)
    # dkp = []
    keep = np.ones((keypoints.shape[1],)).astype('bool')
    for j in range(keypoints.shape[1]):
        kptd, k = trackKLTRobustly(I_prev, I, keypoints[:,j].T, R_T, KLT_N_ITER, KLT_THRESHOLD)
        dkp[:, j] = kptd
        keep[j] = k
    kpold = keypoints[:, keep]
    kpt_original = kpt_original[:, keep]
    keypoints = keypoints + dkp
    keypoints = keypoints[:, keep]

    keypoints = keypoints.T
    kpold = kpold.T
    kpt_original = kpt_original.T

    return keypoints, kpold, kpt_original

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
    keypoints = keypoints.T
    dkp = np.zeros_like(keypoints)
    # dkp = []
    keep = np.ones((keypoints.shape[1],)).astype('bool')
    for j in range(keypoints.shape[1]):
        kptd, k = trackKLTRobustly(I_prev, I, keypoints[:,j].T, R_T, KLT_N_ITER, KLT_THRESHOLD)
        dkp[:, j] = kptd
        keep[j] = k
    # kpold = keypoints[:, keep]
    keypoints = keypoints + dkp
    keypoints = keypoints[:, keep]
    keypoints = keypoints.T
    return keypoints