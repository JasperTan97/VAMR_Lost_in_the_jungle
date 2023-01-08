import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple

from scipy.spatial import distance_matrix

from code.KLT_main import *
from code.get_relative_pose import get_relative_pose
from code.SIFT_main import SIFT
from code.pnp_ransac_main import PnPransacCV, ransacLocalization
from code.triangulate_new import TriangulateNew
import os

# Constants for tunable parameters
from code.constants import *

# For reading the dataset from file
from glob import glob

DATASET = 'parking'
if DATASET=='parking':
    DS_PATH = './data/parking/images/'
    K_PATH = './data/parking/K.txt'
    # Get K (hardcoded to dataset)
    K = np.array(
        [[331.37, 0,       320,],
         [0,      369.568, 240,],
         [0,      0,       1]])
elif DATASET=='kitti':
    DS_PATH = './data/kitti/05/image_0/'
    K_PATH = './data/parking/K.txt'
    # Get K (hardcoded to dataset)
    K = np.array(
        [[7.188560000000e+02, 0, 6.071928000000e+02],
         [0, 7.188560000000e+02, 1.852157000000e+02],
         [0, 0, 1]])
elif DATASET=='malaga':
    DS_PATH = './data/malaga-urban-dataset-extract-07/malaga-urban-dataset-extract-07_rectified_800x600_Images/'
    K_PATH = './data/parking/K.txt'
    # Get K (hardcoded to dataset)
    K = np.array(
        [[621.18428, 0, 404.0076],
        [0, 621.18428, 309.05989],
        [0, 0, 1]])
else:
    raise ValueError(f"Unrecognised dataset {DATASET}. Valid options are \'kitti\', \'parking\', \'malaga\'")

# Malaga dataset contains left and right images
if DATASET == 'malaga':
    DS_GLOB = sorted( glob(DS_PATH+'*left.jpg') )
else:
    DS_GLOB = sorted( glob(DS_PATH+'*.png') )

Pose = np.ndarray
class VO_state:
    '''
    State that contains structs for
    - Point correspondences
        - P (2d coords (u,v,1) homogenous point correspondences)
        - X (3d coords (x,y,z,1) homogenous point positions)
    - Candidate points
        - C (2d coords (u,v) candidate keypoints)
        - F (2d coords (u,v) of first observation of candidate keypoints )
        - T (R,T of transform from World to frame where candidate keypoints were first observed)
    - Methods
        - As long as we insert X in the same order as P, and F,T in the same order as C, there should be no mistakes
    '''
    def __init__(self,
                P: np.ndarray=np.empty((0,2))) -> None:
        self.P = P

def feature_detect_describe(image, detector="SIFT", descriptor=None) -> Tuple[np.ndarray, np.ndarray]: # returns locations, descriptions
    '''
    Given an image, returns:
    - Keypoint locations (N x 2 ndarray)
    - Keypoint descriptors (N x K ndarray, where K is the dimensionality of the descriptor chosen)

    Valid Detectors:
    - "SIFT"
    - "harris"

    Valid Descriptors:
    - Choosing "SIFT" returns SIFT descriptors
    - None
    '''

    if detector == "SIFT":
        return SIFT(image, ROTATION_INVARIANT, CONTRAST_THRESHOLD, RESCALE_FACTOR, SIFT_SIGMA, NUM_SCALES, NUM_OCTAVES)
    elif detector == "harris":
        # Dst is the response map from Harris detector
        dst = cv2.cornerHarris(image, HARRIS_BLOCK_SIZE, HARRIS_K_SIZE, HARRIS_K)

        # Select the N_KPTS best keypoints
        kps = np.zeros((HARRIS_N_KPTS, 2))
        r = HARRIS_R
        temp_scores = np.pad(dst, [(r, r), (r, r)], mode='constant', constant_values=0)

        for i in range(HARRIS_N_KPTS):
            kp = np.unravel_index(temp_scores.argmax(), temp_scores.shape)
            kps[i, :] = np.array(kp) - r
            temp_scores[(kp[0] - r):(kp[0] + r + 1), (kp[1] - r):(kp[1] + r + 1)] = 0
    elif detector == 'good':
        detector = cv2.GFTTDetector_create()
        features = detector.detect(image)
        features = cv2.KeyPoint_convert(sorted(features, key = lambda p: p.response, reverse=True))
        return features, None

    if descriptor is None:
        kps[:, [1, 0]] = kps[:, [0, 1]]
        return kps, None

    else:
        raise NotImplementedError(f"Descriptor {descriptor} has not been implemented.")

def initialiseVO(I) -> VO_state:
    '''
    Bootstrapping
    '''

    kp0, _ = feature_detect_describe(I[0], detector="harris")

    # 2. Feature matching between I1, I0 features
    #    To obtain feature correspondences P0
    kp1, kp0 = KLT_bootstrapping_CV2(kp0, kp0, I[1], I[0])

    for i in range(len(I)-2):
        kp1, kp0 = KLT_bootstrapping_CV2(kp1, kp0, I[i+2], I[i+1])

    # 3. Get Fundemental matrix
    # kp0 and kp1 are Nx2 Numpy arrays containing the pixel coords of the matches.
    F, _ = cv2.findFundamentalMat(
        kp0, kp1,
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        confidence=RANSAC_PROB_SUCCESS,
        maxIters=RANSAC_NUM_ITS
    )

    # Essential Matrix
    E = K.T @ F @ K
    # Convert points format for OpenCV into 3xN homogenous pixel coordinates
    points0 = np.vstack([kp0.T, np.ones((1, kp0.shape[0]))])
    points1 = np.vstack([kp1.T, np.ones((1, kp1.shape[0]))])

    # Get R, T, X out of E matrix
    # X is a byproduct of disambiguating the possibilities in R, T
    R, T, X0 = get_relative_pose(
        points0, points1, E, K
    )
    return VO_state(P=points0)

def processFrame(I1, I0, S0: VO_state) -> Tuple[VO_state, Pose, Pose]:
    '''
    Continuous VO
    Inputs: Current image I1, previous state S0
    Outputs: Current state S1, current pose T1_WC

    State unpacks to P,X,C,F,T, see VO_state class
    '''

    P0 = S0.P   # unpack state i-1
    P1, inliers = KLT_CV2(P0.T[:,:-1].astype(np.float32), I1, I0) # tracks P0 features in I1

    prev_pts = np.copy(P0.T[inliers,:2].astype(np.float32))
    curr_pts = np.copy(P1)
    E, mask = cv2.findEssentialMat(curr_pts.astype(np.float32), prev_pts.astype(np.float32), K, cv2.RANSAC, 0.99, 1.0, None)
    prev_pts = np.array([pt for (idx, pt) in enumerate(prev_pts) if mask[idx] == 1])
    curr_pts = np.array([pt for (idx, pt) in enumerate(curr_pts) if mask[idx] == 1])
    _, R_CW, t_CW, _ = cv2.recoverPose(E, curr_pts, prev_pts, K)
    print(f"{len(curr_pts):04d} features left after pose estimation.")

    T1_CW = np.hstack((R_CW, t_CW)) # Get transformation matrix in SE3 (without bottom row)
    # Add new features to keep C1 from shrinking
    candi_kp, _ = feature_detect_describe(I1, detector="harris")
    l_p = distance_matrix(candi_kp, P1).min(axis=1)<4
    candi_kp = candi_kp[~l_p,:]
    P1 = np.vstack([P1, candi_kp])
    P1 = np.vstack([P1.T, np.ones(P1.shape[0])])
    S1 = VO_state(P=P1) # repack state i

    return S1, T1_CW

def readGroundtuthPosition(frameId):
    groundtruthFile = os.path.join("./data/kitti/poses/", "05.txt")
    with open(groundtruthFile) as f:
        lines = f.readlines()

        _, _, _, tx, _, _, _, ty, _, _, _, tz = list(map(float, lines[frameId].rstrip().split(" ")))
        _, _, _, tx_prev, _, _, _, ty_prev, _, _, _, tz_prev = list(map(float, lines[frameId-1].rstrip().split(" ")))

        position = (tx, ty, tz)
        scale = np.sqrt((tx-tx_prev)**2 + (ty-ty_prev)**2  + (tz-tz_prev)**2)

        return position, scale

def main() -> None:
    np.set_printoptions(precision=3, suppress=True)

    # Bootstrap upto manually selected BOOTSTRAP_FRAME
    I = []
    for img_idx, img_path in enumerate(DS_GLOB[STARTING_FRAME:]):
        if img_idx <= BOOTSTRAP_FRAME:
            I0 = cv2.imread( img_path )
            img0_gray = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
            I.append(img0_gray)
    I0 = I[0]

    bootstrapped_state = initialiseVO(I)

    T0 = np.hstack((np.eye(3), np.zeros((3,1)))) # identity SE3 member for initial pose to signify world frame
    cameraPose = T0[:,3]
    cameraRot = T0[:,:3]
    print("Bootstrap done")
    # Continuous VO
    prev_state = bootstrapped_state # use bootstrapped state as first state
    prev_frame = I[0]

    y = []      # For the global trajectory
    x = []
    dy = []
    dx = []
    num_tracked_kps = []

    # For triangulating
    triang_pose = [T0]                              # incremenetal [R|T] wrt World frame (identity)
    triang_kps = [bootstrapped_state.P[:2,:].T]     # Keypoints detected (Nx2 array)
    img_hist = [img0_gray]                          # Images

    T_C1C2 = None
    plt.figure(figsize=[12,7])

    for img_idx, img_path in enumerate(DS_GLOB[STARTING_FRAME:]):
        # If we want to attempt to triangulate between frames to obtain a pointcloud
        plot_pointcloud = True
        if plot_pointcloud:
            ax0 = plt.subplot(2, 2, 1)
            ax0.axis('equal')                   # Plots trajectory
            ax1 = plt.subplot(2, 2, 2)          # Plots image with keypoints
            ax2 = plt.subplot(2, 2, 3)          # Plots history of num tracked keypoints for last 20 frames
            ax3 = plt.subplot(2, 2, 4)          # Trajectory of last 20 frames and landmarks
            ax3.axis('equal')
        else:
            ax0 = plt.subplot(2, 2, 3)
            ax0.axis('equal')                   # Plots trajectory
            ax1 = plt.subplot(2, 2, (1,2))      # Plots image with keypoints
            ax2 = plt.subplot(2, 2, 4)          # Plots history of num tracked keypoints for last 20 frames

        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        state, T_C1C2 = processFrame(frame, prev_frame, prev_state) # continuous VO markov chain

        prev_state = state # set current state to previous state for next image
        prev_frame = frame

        cameraPose = cameraPose + cameraRot.dot(T_C1C2[:,3])
        cameraRot = T_C1C2[:,:3].dot(cameraRot)

        ## PLOTTING ##
        x.append(cameraPose[0])
        y.append(cameraPose[2])
        dx.append(cameraRot[0,2])
        dy.append(cameraRot[2,2])
        num_tracked_kps.append(state.P.shape[1])

        if len(y) > 20:     # Tracking only the last 20 frames
            num_tracked_kps.pop(0)

        # Plot trajectory
        ax0.clear()
        ax0.set_title(f"Current pose: (x: {cameraPose[0]:06.2f}, y:{cameraPose[2]:06.2f})")
        ax0.scatter(x, y, marker='.', s=1, color='black')
        ax0.plot(cameraPose[0], cameraPose[2], marker='.', color='red')

        # Plot tracking of keypoints
        ax1.clear()
        ax1.set_title(f"FRAME: {img_idx + STARTING_FRAME}")
        ax1.imshow(frame, cmap='gray', vmin=0, vmax=255)
        ax1.scatter(state.P[ 0,:], state.P[1,:], marker='x', linewidths=0.5, color='red')

        # number of tracked landmarks over the past 20 frames
        ax2.clear()
        ax2.plot(num_tracked_kps, 'kx--')
        ax2.set_title(f"No. KPs tracked in past {len(num_tracked_kps)} frames")
        ax2.set_ylim(0, 5000)

        # Trajectory of last 20 frames and landmarks
        if plot_pointcloud:
            # Triangulate keypoints every frame
            triang_pose.append(np.hstack((cameraRot,cameraPose.reshape(-1,1))))
            triang_kps.append(state.P[:2,:].T)
            img_hist.append(frame)

            if len(triang_pose) > BOOTSTRAP_FRAME:
                # Track features from 1st frame through to the end
                kp0 = triang_kps[0]
                kp1, kp0 = KLT_bootstrapping_CV2(kp0, kp0, img_hist[1], img_hist[0])
                for i in range(len(img_hist)-2):
                    kp1, kp0 = KLT_bootstrapping_CV2(kp1, kp0, img_hist[i+2], img_hist[i+1])

                F, _ = cv2.findFundamentalMat(
                    kp0, kp1,
                    method=cv2.FM_RANSAC,
                    ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
                    confidence=RANSAC_PROB_SUCCESS,
                    maxIters=RANSAC_NUM_ITS
                )

                E = K.T @ F @ K     # Essential Matrix
                # Convert points format for OpenCV into 3xN homogenous pixel coordinates
                points0 = np.vstack([kp0.T, np.ones((1, kp0.shape[0]))])
                points1 = np.vstack([kp1.T, np.ones((1, kp1.shape[0]))])

                _, _, P_W = get_relative_pose(
                    points0, points1, E, K
                )
                # Filter P_W for those that are in front of camera
                P_W_sel = P_W[2,:] > 0
                print(f"main_num_points_infront {np.sum(P_W_sel)}")
                P_W = P_W[:, P_W_sel]
                kp1 = kp1[P_W_sel, :]

                # Check parallax
                n_pts = np.sum(P_W_sel)     # number of points in front of camera
                v0 = -(np.hstack(n_pts*[triang_pose[0][:,-1].reshape(-1,1)]) - P_W[:3,:])
                v1 = -(np.hstack(n_pts*[triang_pose[-1][:,-1].reshape(-1,1)]) - P_W[:3,:])
                v0 = v0/np.linalg.norm(v0, axis=0)
                v1 = v1/np.linalg.norm(v1, axis=0)
                cosalpha = (v0*v1).sum(axis=0)
                pts_bool = np.logical_and(np.round(cosalpha,3) >= 0, cosalpha < PAR_THRESHOLD)
                print(f"Max angle: {np.degrees(np.arccos(np.min(cosalpha))):.3f}, Min angle: {np.degrees(np.arccos(np.max(cosalpha))):.3f}")
                print(f"Distance between frames: {triang_pose[0][:,-1]}, {triang_pose[-1][:,-1]}, {np.linalg.norm(triang_pose[0][:,-1]-triang_pose[-1][:,-1]):.2f}")

                P_W = P_W[:, pts_bool]
                kp1 = kp1[pts_bool, :]
                print(f"main num_points after parallax check {np.sum(pts_bool)}. ")

                if np.sum(pts_bool) > 4:
                    # Ensure P_W is in world frame
                    P_W = triang_pose[0] @ P_W

                    # run 2d-3d correspondences for an estimate of our current position
                    R_CW, t_CW, inliers_pnp = \
                        PnPransacCV(kp1.astype(np.float32),
                            P_W.T.astype(np.float64), K) # Get current pose with RANSAC

                    R_WC = R_CW.T
                    t_WC = -R_CW.T @ t_CW

                    # Compare global pose (t_WC) with pose from incremental estimation
                    method_error = np.linalg.norm(t_WC[[0,2],0] - triang_pose[0][[0,2],3])
                else:
                    method_error = None

                # Pop stuff off the end of the buffer
                triang_kps.pop(0)
                triang_pose.pop(0)
                img_hist.pop(0)
            else:   # If not enough elems in the buffer for bootstrapping
                P_W = None
                method_error = None

            ax3.clear()
            # ax3.set_xlim(x[-1]-100, x[-1]+100)
            # ax3.set_ylim(y[-1]-100, y[-1]+100)
            if P_W is not None:
                ax3.scatter(P_W[0,:], P_W[2,:], marker='x', color='green')
                ax3.arrow(x=t_WC[0,0], y=t_WC[2,0], dx=R_WC[0,2]*15, dy=R_WC[2,2]*15, width=2, color='magenta')
                ax3.arrow(x=triang_pose[0][0,3],
                    y=triang_pose[0][2,3],
                    dx=triang_pose[0][0,2]*10,
                    dy=triang_pose[0][2,2]*10, width=2, color='red')
            for i in range(len(x)-1):
                ax3.arrow(x=x[i], y=y[i], dx=dx[i]*10, dy=dy[i]*10, width=1)
            ax3.arrow(x=x[-1], y=y[-1], dx=dx[-1]*15, dy=dy[-1]*15, width=2, color='yellow')
            if method_error is not None:
                title = f"Last {len(x)} frames. Diff between 3D-2D and 2D-2D: {method_error:.2f}"
            else:
                title = f"Last {len(x)} frames"
            ax3.set_title(title)

        plt.tight_layout()
        plt.pause(0.0001)   # For time to plot

if __name__ == "__main__":
    main()