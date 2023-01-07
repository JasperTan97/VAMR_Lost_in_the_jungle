import numpy as np
from typing import Tuple

from code.triangulation import triangulation

def get_relative_pose(
    points0:np.ndarray,
    points1:np.ndarray,
    E:np.ndarray,
    K:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an essential matrix, compute the camera motion, i.e.,  R and T such
    that E ~ T_x R.
    Then disambiguates from the four possible configurations.

    Input:
    - points1   -  3xN homogeneous coordinates of point correspondences in image 1
    - points2   -  3xN homogeneous coordinates of point correspondences in image 2
    - E(3,3) : 3x3 Essential matrix,
    - K : 3x3 Calibration matrix

    Returns:
    - R -  3x3 the correct rotation matrix
    - T -  3x1 the correct translation vector
    - X - 4xN 3D correspondences in the world (img0) frame
    """

    # Disambiguate E first
    u, _, vh = np.linalg.svd(E)

    # Translation
    u3 = u[:, 2]

    # Rotations
    W = np.array([ [0, -1,  0],
                   [1,  0,  0],
                   [0,  0,  1]])

    R0 = u @ W @ vh
    R1 = u @ W.T @ vh

    if np.linalg.det(R0) < 0:
        R0 *= -1
    if np.linalg.det(R1) < 0:
        R1 *= -1
    Rots = np.stack((R0, R1), axis=-1)

    # We don't know the scale of u3 so we just normalise to 1.
    # Check for norm==0 to prevent a div by 0
    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)

    M1 = K @ np.eye(3,4)

    best_num_pts = 0
    for iRot in range(2):
        R_C2_C1_test = Rots[:,:,iRot]
        for signT in [-1, 1]:
            T_C2_C1_test = u3[:,np.newaxis] * signT
            M2 = K @ np.hstack([R_C2_C1_test, T_C2_C1_test])
            P_C1,_ = triangulation(points0, points1, M1, M2)

            # project in both cameras
            P_C2 = np.hstack([R_C2_C1_test, T_C2_C1_test]) @ P_C1

            num_points_in_front1 = np.sum(P_C1[2,:] > 0)
            num_points_in_front2 = np.sum(P_C2[2,:] > 0)
            total_points_in_front = num_points_in_front1 + num_points_in_front2

            if (total_points_in_front > best_num_pts):
                # Keep the rotation that gives the highest number of points
                # in front of both cameras
                R = R_C2_C1_test
                T = T_C2_C1_test
                best_num_pts = total_points_in_front

                X = P_C1

    return R, T, X
