import numpy as np
from typing import List

from code.constants import RANSAC_REPROJ_THRESHOLD, RANSAC_NUM_ITS
from code.utils import normalise_2d_pts

def findFundementalMat(
    points1:np.ndarray,
    points2:np.ndarray,
    K1: None,
    K2: None
) -> np.ndarray:
    '''
    Find the fundemental matrix using the 8-point algorithm.

    Inputs:
    - points1: 3xN NDArray containing the px coordinates of feature correspondences from image 1
    - points2: 3xN NDArray containing the px coordinates of feature correspondences from image 2

    Optionally, include
    - K1, K2: 3x3 calibration matrices for camera 1 and 2
    If K1, K2 are included, then the essential matrix E is obtained.
    '''
    assert(points1.shape==points2.shape), "Points 1 and 2 should have the same dimension."
    assert(points1.shape[0]==3), "Points should have 3 cols (homogenous pixel coords)"

    # Normalize points
    points1_tilde, T1 = normalise_2d_pts(points1)
    points2_tilde, T2 = normalise_2d_pts(points2)

    n_pts = points1.shape[1]
    assert(n_pts>=8), \
            'Insufficient number of points to compute fundamental matrix (need >=8)'

    # Compute the measurement matrix A of the linear homogeneous system whose
    # solution is the vector representing the fundamental matrix.
    A = np.zeros((n_pts,9))
    for i in range(n_pts):
        A[i,:] = np.kron( points1_tilde[:,i], points2_tilde[:,i] ).T

    # "Solve" the linear homogeneous system of equations A*f = 0.
    # The correspondences x1,x2 are exact <=> rank(A)=8 -> there exist an exact solution
    # If measurements are noisy, then rank(A)=9 => there is no exact solution, 
    # seek a least-squares solution.
    _, _, vh= np.linalg.svd(A, full_matrices = False)
    F = np.reshape(vh[-1,:], (3,3)).T

    # Enforce det(F)=0 by projecting F onto the set of 3x3 singular matrices
    u, s, vh = np.linalg.svd(F)
    s[2] = 0
    F = u @ np.diag(s) @ vh

    # Undo the normalization
    F = T2.T @ F @ T1

    # Extract E if K is provided
    if K1 is not None and K2 is not None:
        assert (K1.shape==(3,3)), "K1 should be a 3x3 camera calibration matrix"
        assert (K2.shape==(3,3)), "K2 should be a 3x3 camera calibration matrix"

        F = np.linalg.inv(K2.T) @ F @ np.linalg.inv(K1)

    return F


def findFundementalMatRANSAC(
    points1:np.ndarray,
    points2:np.ndarray,
    K1: None,
    K2: None
) -> np.ndarray:
    '''
    Find the fundemental matrix using the 8-point algorithm.

    Inputs:
    - points1: 3xN NDArray containing the px coordinates of feature correspondences from image 1
    - points2: 3xN NDArray containing the px coordinates of feature correspondences from image 2

    Optionally, include
    - K1, K2: 3x3 calibration matrices for camera 1 and 2
    If K1, K2 are included, then the essential matrix E is obtained.
    '''

    # TODO . This is done by calling the CV2 function...

    # Idea: Do RANSAC in parallel
    for _ in range(RANSAC_NUM_ITS):
        # Select 8 points at random
        np.random.choice(len())
        # Compute 8-point algo with these sampled points
        # For each datapoint:
            # Calculate the residual for each datapoint
            # Select datapoints in good_matches that support the hypothesis
    # Select the set with the max number of inliers
    # Calculate model parameteres again with all inliers
    # Obtain R, T from essential matrix.

        pass