import cv2
import numpy as np
import matplotlib.pyplot as plt

from code.track_klt_robustly import trackKLTRobustly
from code.constants import *

def KLT_bootstrapping_CV2(keypoints, kpt_original, I, I_prev):
    keypoints = keypoints.astype(np.float32)
    lk_params = dict( winSize  = (R_T, R_T),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, KLT_N_ITER, KLT_THRESHOLD))
    p1, st, err = cv2.calcOpticalFlowPyrLK(I_prev, I, keypoints, None, **lk_params)
    #print(kpt_original.shape, st.shape)
    st = st.reshape(keypoints.shape[0])
    kpt_original = kpt_original[st==1,:]
    #kpold = keypoints[st==1,:]
    keypoints = p1[st==1,:]

    return keypoints, kpt_original

def KLT_CV2(keypoints, I, I_prev):
    keypoints = keypoints.astype(np.float32)
    lk_params = dict( winSize  = (R_T, R_T),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, KLT_N_ITER, KLT_THRESHOLD))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(I_prev, I, keypoints, None, **lk_params)
    # kpold = keypoints[st==1]
    st = st.reshape(keypoints.shape[0])
    keypoints = p1[st==1,:]

    return keypoints, st

def KLT_bootstraping(keypoints, kpt_original, I, I_prev):
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