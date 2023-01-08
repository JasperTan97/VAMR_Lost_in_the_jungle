import cv2
import numpy as np
import matplotlib.pyplot as plt
from code.constants import *

def PnPransacCV(P1, X1, K):
    _, rvec, tvec, inliers = cv2.solvePnPRansac(X1, P1, K, distCoeffs=None)
    # rvec1, tvec1 = cv2.solvePnPRefineLM(X1, P1, K, distCoeffs=None)
    RotMat = cv2.Rodrigues(rvec)[0]

    return RotMat, tvec, inliers

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
        # Check the inlier count
        idx_inlier = errors < PIXEL_TOLERANCE**2
        n_in = np.count_nonzero(idx_inlier)
        if n_in > N_inliers:
            # Resolve the PnP problem with the inliers alone to get a better estimate
            N_inliers = n_in
            _, R1_best, T1_best = cv2.solvePnP(objectPoints=X1[:,idx_inlier],
                                               imagePoints=P1[:,idx_inlier],
                                               cameraMatrix=K,
                                               distCoeffs=None)
    
    return R1_best, T1_best


def ransacLocalization(matched_query_keypoints, corresponding_landmarks, K):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.
    """
    use_p3p = True
    tweaked_for_more = True
    adaptive = True   # whether or not to use ransac adaptively

    if use_p3p:
        num_iterations = 1000 if tweaked_for_more else 200
        pixel_tolerance = 10
        k = 3
    else:
        num_iterations = 2000
        pixel_tolerance = 10
        k = 6

    if adaptive:
        num_iterations = float('inf')

    # Initialize RANSAC

    best_inlier_mask = np.zeros(matched_query_keypoints.shape[1])
    # (row, col) to (u, v)
    # matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)
    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # RANSAC
    i = 0
    while num_iterations > i:
        # Model from k samples (DLT or P3P)
        indices = np.random.permutation(corresponding_landmarks.shape[0])[:k]
        landmark_sample = corresponding_landmarks[indices, :]
        keypoint_sample = matched_query_keypoints[:, indices]

        if use_p3p:
            success, rotation_vectors, translation_vectors = cv2.solveP3P(landmark_sample, keypoint_sample.T, K,
                                                                        None, flags=cv2.SOLVEPNP_P3P)
            t_C_W_guess = []
            R_C_W_guess = []
            for rotation_vector in rotation_vectors:
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                for translation_vector in translation_vectors:
                    R_C_W_guess.append(rotation_matrix)
                    t_C_W_guess.append(translation_vector)

        else:
            M_C_W_guess = estimatePoseDLT(keypoint_sample.T, landmark_sample, K)
            R_C_W_guess = M_C_W_guess[:, :3]
            t_C_W_guess = M_C_W_guess[:, -1]

        # Count inliers
        if not use_p3p:
            C_landmarks = np.matmul(R_C_W_guess, corresponding_landmarks[:, :, None]).squeeze(-1) + t_C_W_guess[None, :]
            projected_points = projectPoints(C_landmarks, K)
            difference = matched_query_keypoints - projected_points.T
            errors = (difference**2).sum(0)
            is_inlier = errors < pixel_tolerance**2

        else:
            # If we use p3p, also consider inliers for the 4 solutions.
            is_inlier = np.zeros(corresponding_landmarks.shape[0])
            for alt_idx in range(len(R_C_W_guess)):
                C_landmarks = np.matmul(R_C_W_guess[alt_idx], corresponding_landmarks[:, :, None]).squeeze(-1) + \
                              t_C_W_guess[alt_idx][None, :].squeeze(-1)
                projected_points = projectPoints(C_landmarks, K)
                difference = matched_query_keypoints - projected_points.T
                errors = (difference ** 2).sum(0)
                alternative_is_inlier = errors < pixel_tolerance ** 2
                if alternative_is_inlier.sum() > is_inlier.sum():
                    is_inlier = alternative_is_inlier

        min_inlier_count = 30 if tweaked_for_more else 6

        if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= min_inlier_count:
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        if adaptive:
            # estimate of the outlier ratio
            outlier_ratio = 1 - max_num_inliers / is_inlier.shape[0]
            # formula to compute number of iterations from estimated outlier ratio
            confidence = 0.95
            upper_bound_on_outlier_ratio = 0.90
            outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
            num_iterations = np.log(1-confidence)/np.log(1-(1-outlier_ratio)**k)
            # cap the number of iterations at 15000
            num_iterations = min(15000, num_iterations)

        num_iteration_history.append(num_iterations)
        max_num_inliers_history.append(max_num_inliers)

        i += 1
    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        M_C_W  = estimatePoseDLT(matched_query_keypoints[:, best_inlier_mask].T, corresponding_landmarks[best_inlier_mask, :], K)
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]

        # if adaptive:
        #     print("    Adaptive RANSAC: Needed {} iteration to converge.".format(i - 1))
        #     print("    Adaptive RANSAC: Estimated Ouliers: {} %".format(100 * outlier_ratio))

    return R_C_W, t_C_W, best_inlier_mask, max_num_inliers_history, num_iteration_history

def projectPoints(points_3d, K, D=np.zeros([4, 1])):
    """
    Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
    distortion coefficients (4x1).
    """
    # get image coordinates
    projected_points = np.matmul(K, points_3d[:, :, None]).squeeze(-1)
    projected_points /= projected_points[:, 2, None]

    # apply distortion
    projected_points = distortPoints(projected_points[:, :2], D, K)

    return projected_points

def distortPoints(x, D, K):
    """Applies lens distortion D(2) to 2D points x(Nx2) on the image plane. """

    k1, k2 = D[0], D[1]

    u0 = K[0, 2]
    v0 = K[1, 2]

    xp = x[:, 0] - u0
    yp = x[:, 1] - v0

    r2 = xp**2 + yp**2
    xpp = u0 + xp * (1 + k1*r2 + k2*r2**2)
    ypp = v0 + yp * (1 + k1*r2 + k2*r2**2)

    x_d = np.stack([xpp, ypp], axis=-1)

    return x_d

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    
    # Convert 2D to normalized coordinates
    p_norm = (np.linalg.inv(K) @ np.c_[p, np.ones((p.shape[0], 1))].T).T

    # Build measurement matrix Q
    num_corners = p_norm.shape[0]
    Q = np.zeros((2*num_corners, 12))

    for i in range(num_corners):
        u = p_norm[i, 0]
        v = p_norm[i, 1]

        Q[2*i, 0:3] = P[i,:]
        Q[2*i, 3] = 1
        Q[2*i, 8:11] = -u * P[i,:]
        Q[2*i, 11] = -u
        
        Q[2*i+1, 4:7] = P[i,:]
        Q[2*i+1, 7] = 1
        Q[2*i+1, 8:11] = -v * P[i,:]
        Q[2*i+1, 11] = -v

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    u, s, v = np.linalg.svd(Q, full_matrices=True)
    M_tilde = np.reshape(v.T[:,-1], (3,4));
    
    # Extract [R | t] with the correct scale
    if (np.linalg.det(M_tilde[:, :3]) < 0):
        M_tilde *= -1

    R = M_tilde[:, :3]

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    u, s, v = np.linalg.svd(R);
    R_tilde = u @ v;

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = np.linalg.norm(R_tilde, 'fro')/np.linalg.norm(R, 'fro');

    # Build M_tilde with the corrected rotation and scale
    M_tilde = np.c_[R_tilde, alpha * M_tilde[:,3]];
    
    return M_tilde