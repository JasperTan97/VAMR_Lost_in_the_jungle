import numpy as np
from code.utils import vec_to_cross_mat

def triangulation(
    points1: np.ndarray,
    points2: np.ndarray,
    M1: np.ndarray,
    M2: np.ndarray,
) -> np.ndarray:
    '''
    Inputs:
    - points1
    - points2: 3xN NDArray containing homogenous 2d-2d correspondences
    - M1
    - M2: 3x4 Projection matrices [R|T]
    Returns:
    - P: 4xN matrix containing the triangulated points in homogenous coordinates.
    '''
    assert(points1.shape==points2.shape), "Input correspondences should have the same dimension (3 rows, N cols)"
    assert(M1.shape==(3,4)), "M1 should be (3, 4)"
    assert(M2.shape==(3,4)), "M2 should be (3, 4)"

    n_pts = points1.shape[1]
    P = np.zeros((4, n_pts))

    for i in range(n_pts):
        A1 = vec_to_cross_mat(points1[:,i]) @ M1
        A2 = vec_to_cross_mat(points2[:,i]) @ M2

        # Stack
        A = np.vstack([A1, A2])

        # Solve
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        P[:,i] = vh.T[:,-1]     # Last column of V

    # Dehomogenize
    # P_bool = P[3,:] > 0
    P /= P[3,:]
    # P_bool = P[2,:] > 0

    return P