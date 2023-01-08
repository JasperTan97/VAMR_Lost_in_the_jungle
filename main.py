import cv2
import numpy as np
import matplotlib.pyplot as plt


from typing import List, Tuple, Dict

#from ismember import ismember
from scipy.spatial import distance_matrix

from code.KLT_main import *
from code.get_relative_pose import get_relative_pose
from code.SIFT_main import SIFT
from code.pnp_ransac_main import PnPransacCV, ransacLocalization
from code.triangulate_new import TriangulateNew
import os
# from code.linear_triangulation import linearTriangulation # (removed as unnecesary)

# Constants for tunable parameters
from code.constants import *

# For reading the dataset from file
from glob import glob

DATASET = 'kitti'
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
    raise ValueError

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
                P: np.ndarray=np.empty((0,2)),
                X: np.ndarray=np.empty((0,2)),
                C: np.ndarray=None,
                F: np.ndarray=None,
                T: np.ndarray=None) -> None:
        self.P = P
        self.X = X
        self.C = C
        self.F = F
        self.T = T
        # assert P.shape[-1] == X.shape[-1], f"P (shape {P.shape}) and X (shape {X.shape}) have diff lengths"
        # assert P.shape[0] == 3, "P are the homogenous pixel correspondences, it should have structure (u,v,1)"
        # assert X.shape[0] == 4, "X are the homogenous 3d point correspondences, it should have structure (x,y,z,1)"
        # if C is not None:
        #     assert C.shape[-1] == F.shape[-1]
        #     assert C.shape[-1] == T.shape[-1]

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
    candi_keypoints = np.copy(kp0)

    # for checking
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.imshow(I[0], cmap='gray', vmin=0, vmax=255 )
    # ax.plot(kp0[:, 0], kp0[:, 1], 'rx')
    # plt.pause(2)
    

    # 2. Feature matching between I1, I0 features
    #    To obtain feature correspondences P0
    kp1, kp0 = KLT_bootstrapping_CV2(kp0, kp0, I[1], I[0])

    for i in range(len(I)-2):
        kp1, kp0 = KLT_bootstrapping_CV2(kp1, kp0, I[i+2], I[i+1])
        # for checking
        # plt.clf()
        # plt.close(fig)
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        
        #print(kp1[0:10])
       
        # ax.imshow(I[i+2], cmap='gray', vmin=0, vmax=255)
        # keypoints_ud = np.flipud(kp1).T
        # kpold_ud = np.flipud(kp0).T
        # print(keypoints_ud.shape, kpold_ud.shape)
        # x_from = keypoints_ud[0, :]
        # x_to = kpold_ud[0,:]
        # y_from = keypoints_ud[1, :]
        # y_to = kpold_ud[1,:]
        # ax.plot(np.r_[x_from[np.newaxis, :], x_to[np.newaxis,:]], 
        #         np.r_[y_from[np.newaxis,:], y_to[np.newaxis,:]], 'g-',
        #         linewidth=3)
    #     ax.plot(kp1[:, 0], kp1[:, 1], 'rx')
    #     print(kp1.shape)
    #     ax.set_xlim([0, I[i+2].shape[1]])
    #     ax.set_ylim([I[i+2].shape[0], 0])
    #     plt.pause(0.5)
    # plt.show()

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
    # print(E.shape)
    # Convert points format for OpenCV into 3xN homogenous pixel coordinates
    points0 = np.vstack([kp0.T, np.ones((1, kp0.shape[0]))])
    points1 = np.vstack([kp1.T, np.ones((1, kp1.shape[0]))])

    check = np.linalg.norm( np.sum(points1 * (F @ points0)) ) / np.sqrt(points0.shape[1])
    # Get R, T, X out of E matrix
    # X is a byproduct of disambiguating the possibilities in R, T
    R, T, X0 = get_relative_pose(
        points0, points1, E, K
    )
    # _, R_check, T_check, _ = cv2.recoverPose(E, kp0, kp1)
    # print(R_check, '\n', R)
    # print(T_check,'\n', T)

    # check_pts = K @ np.hstack([R, T]) @ X0
    check_pts = K @ np.eye(3,4) @ X0
    check_pts /= check_pts[2,:]

    diff = check_pts[:2,:]-kp0.T
    reprojError = np.mean(np.linalg.norm(diff, axis=0))
    # print(reprojError)

    # Initialize C, F, T
    candi_kp, _ = feature_detect_describe(I[0], detector="harris") # All features in the Bootstrap frame
    # Remove features already triangulated
    dist_mat = distance_matrix(candi_kp, kp0)
    l = dist_mat.min(axis=1)<2
    candi_kp = candi_kp[~l,:]
    # ax.imshow(I[BOOTSTRAP_FRAME], cmap='gray', vmin=0, vmax=255)
    # ax.plot(kp1[:, 0], kp1[:, 1], 'rx')
    # ax.plot(candi_kp[~l, 0], candi_kp[~l, 1], 'gx')
    # plt.show()
    C0 = np.vstack([candi_kp.T, np.ones((1, candi_kp.shape[0]))])
    F0 = np.vstack([candi_kp.T, np.ones((1, candi_kp.shape[0]))])
    pose_flattened = np.hstack([np.eye(3).flatten(), np.zeros((3,1)).flatten()])
    # pose_flattened = np.eye(3,4).flatten()
    T0 = np.hstack([pose_flattened.reshape(-1,1)]*C0.shape[-1])

    # 4. Bundle adjustment to refine R, T, X0
    # TODO figure this out

    return VO_state(P=points0, X=X0, C=C0, F=F0, T=T0)

def processFrame(I1, I0, S0: VO_state) -> Tuple[VO_state, Pose, Pose]:
    '''
    Continuous VO
    Inputs: Current image I1, previous state S0
    Outputs: Current state S1, current pose T1_WC

    State unpacks to P,X,C,F,T, see VO_state class
    '''

    P0 = S0.P 
    X0 = S0.X 
    C0 = S0.C 
    F0 = S0.F 
    T0 = S0.T # unpack state i-1

    P1, inliers = KLT_CV2(P0.T[:,:-1].astype(np.float32), I1, I0) # tracks P0 features in I1
    # P1 = P0[inliers, :]
    # X1 = X0[:, inliers.astype(bool)] # Update X1 from P1 and X0
    # print(np.sum(inliers))
    # R_CW, t_CW, inliers_pnp = PnPransacCV(P1.astype(np.float32), X1[:-1,:].T.astype(np.float64), K) # Get current pose with RANSAC

    prev_pts = np.copy(P0.T[inliers,:2].astype(np.float32))
    curr_pts = np.copy(P1)
    E, mask = cv2.findEssentialMat(curr_pts.astype(np.float32), prev_pts.astype(np.float32), K, cv2.RANSAC, 0.99, 1.0, None)
    prev_pts = np.array([pt for (idx, pt) in enumerate(prev_pts) if mask[idx] == 1])
    curr_pts = np.array([pt for (idx, pt) in enumerate(curr_pts) if mask[idx] == 1])
    _, R_CW, t_CW, _ = cv2.recoverPose(E, curr_pts, prev_pts, K)
    print(f"{len(curr_pts)} features left after pose estimation.")

    # R_CW, t_CW, inliers_pnp, _, _ = ransacLocalization(P1.T, X1[:-1,:].T, K)
    # t_CW = t_CW.reshape(-1,1)
    # inliers_pnp = inliers_pnp.reshape(-1)
    # print("Fraction of inliers:",inliers_pnp.shape[0]/P1.shape[0])
    # X1 = X1[:, inliers_pnp]
    pose_flattened = np.hstack([R_CW.flatten(), t_CW.flatten()])
    T1_CW = np.hstack((R_CW, t_CW)) # Get transformation matrix in SE3 (without bottom row)
    # C1, inliers_candidates = KLT_CV2(C0.T[:,:-1].astype(np.float32), I1, I0) # track C0 features in I1
    # Remove features with lost tracking
    # F1 = F0[:, inliers_candidates.astype(bool)]
    # T1 = T0[:, inliers_candidates.astype(bool)]
    # Add new features to keep C1 from shrinking
    candi_kp, _ = feature_detect_describe(I1, detector="harris")
    # l_c = distance_matrix(candi_kp, C1).min(axis=1)<4
    l_p = distance_matrix(candi_kp, P1).min(axis=1)<4
    # l = np.logical_or(l_c,l_p)
    candi_kp = candi_kp[~l_p,:]
    # print(np.sum(l_p))
    # C1 = np.vstack([C1, candi_kp])
    # F1 = np.hstack([F1, np.vstack([candi_kp.T, np.ones(candi_kp.shape[0])])])
    # T1 = np.hstack([T1, np.hstack(candi_kp.shape[0]*[pose_flattened.reshape(-1,1)])])
    # triangulate new points
    # P1, X1, C1, F1, T1 = TriangulateNew(P1, X1, C1, F1, T1, T1_CW, K)
    # F1 = firstObservationTracker(C0, F0, C1) # update F1 with C1
    # T1 = cameraPoseMappedTo_c(C0, T0, C1) # update T1 with C1
    # P1, X1, C1, F1, T1 = TriangulateProMaxPlusUltra(P1, X1, C1, F1, T1) # Add new points to pointcloud in P1, updates X1 accordingly, removes those points from C1, F1, and T1
    P1 = np.vstack([P1, candi_kp])
    P1 = np.vstack([P1.T, np.ones(P1.shape[0])])
    #C1 = np.vstack([C1.T, np.ones(C1.shape[0])])
    S1 = VO_state(P=P1, X=X0, C=C0, F=F0, T=T0) # repack state i

    # T1_CW_check = np.hstack([R_check, T_check])
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

    # Bootstrap
    I = []
    for img_idx, img_path in enumerate(DS_GLOB[STARTING_FRAME:]):
        if img_idx <= BOOTSTRAP_FRAME:
            I0 = cv2.imread( img_path )
            img0_gray = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
            I.append(img0_gray)
    I0 = I[0]

    bootstrapped_state = initialiseVO(I)
    # print(bootstrapped_state.P)
    # print(bootstrapped_state.X)

    T0 = np.hstack((np.eye(3), np.zeros((3,1)))) # identity SE3 member for initial pose to signify world frame
    odom = [T0]
    cameraPose = T0[:,3]
    cameraRot = T0[:,:3]
    print("Bootstrap done")
    # return
    # Continuous VO
    prev_state = bootstrapped_state # use bootstrapped state as first state
    prev_frame = I[0]

    # fig0 = plt.figure(1)
    # fig1 = plt.figure(2)
    # ax0 = fig0.add_subplot()
    # ax1 = fig1.add_subplot()

    y = []
    x = []
    dy = []
    dx = []
    num_tracked_kps = []

    triang_pose = [T0]
    triang_kps = [bootstrapped_state.P[:2,:].T]
    img_hist = [img0_gray]

    T_C1C2 = None
    plt.figure(figsize=[12,7])

    for img_idx, img_path in enumerate(DS_GLOB[STARTING_FRAME:]):
        plot_pointcloud = False
        if plot_pointcloud:
            ax0 = plt.subplot(2, 2, 1)
            ax0.axis('equal')                   # Plots trajectory
            ax1 = plt.subplot(2, 2, 2)          # Plots image with keypoints
            ax2 = plt.subplot(2, 2, 3)          # Plots history of num tracked keypoints for last 20 frames
            ax3 = plt.subplot(2, 2, 4)          # Trajectory of last 20 frames and landmarks
            ax3.axis('equal')
        else:
            ax0 = plt.subplot(2, 2, 3)
            ax0.axis('equal')               # Plots trajectory
            ax1 = plt.subplot(2, 2, (1,2))  # Plots image with keypoints
            ax2 = plt.subplot(2, 2, 4)      # Plots history of num tracked keypoints for last 20 frames

        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #prev_T_C1C2 = T_C1C2
        state, T_C1C2 = processFrame(frame, prev_frame, prev_state) # continuous VO markov chain

        prev_state = state # set current state to previous state for next image
        prev_frame = frame

        # _, scale = readGroundtuthPosition(img_idx)
        scale = 1.0
        cameraPose = cameraPose + scale*cameraRot.dot(T_C1C2[:,3])
        cameraRot = T_C1C2[:,:3].dot(cameraRot)

        # K_invP = np.linalg.inv(K) @ state.P
        # #print(np.linalg.inv(K).shape, state.P.shape, K_invP.shape, np.hstack((cameraRot, cameraPose.reshape(-1,1))).shape)
        # P_W = np.hstack((cameraRot, cameraPose.reshape(-1,1))).T @ K_invP

        #P_W = P_W / P_W[3,:]
        #print(P_W.shape, P_W[:5])
        # n_front = np.sum(X_cam[2,:] > 0)
        # print("Fraction of points infront",n_front/X_cam.shape[1], prev_state.X.shape[1])

        # Convert the inverse
        # R_C_W = T_WC[:3,:3]
        # t_C_W = T_WC[:,-1]
        # t_W_C = -np.matmul(R_C_W.T, t_C_W)
        # if t_W_C_last is not None:
        #     t_rel = t_W_C - t_W_C_last
        #     t_rel_norm = np.linalg.norm(t_rel)
        #     t_W_C = t_W_C_last + t_rel / t_rel_norm
        # t_W_C_last = t_W_C
        # print(prev_state.X.shape[1])

        ## PLOTTING ##

        x.append(cameraPose[0])
        y.append(cameraPose[2])
        dx.append(cameraRot[0,2])
        dy.append(cameraRot[2,2])
        num_tracked_kps.append(state.P.shape[1])

        if len(y) > 20:
            # x.pop(0)
            # y.pop(0)
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
        # P = Tracked Keypoints
        # C = Candidate Keypoints

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
                # Ensure P_W is in world frame
                P_W = triang_pose[0] @ P_W

                # print(kp1.shape, P_W[:,:].T.shape)

                # run 2d-3d correspondences
                R_CW, t_CW, inliers_pnp = \
                    PnPransacCV(kp1.astype(np.float32),
                        P_W[:,:].T.astype(np.float64), K) # Get current pose with RANSAC

                # Compare R_CW with pose
                meth_error = np.linalg.norm(t_CW - cameraPose)

                triang_kps.pop(0)
                triang_pose.pop(0)
                img_hist.pop(0)
            else:
                P_W = None
                meth_error = None

            ax3.clear()
            if P_W is not None:
                ax3.scatter(P_W[0,:], P_W[2,:], marker='x', color='green')
            for i in range(len(x)-1):
                ax3.arrow(x=x[i], y=y[i], dx=dx[i]*10, dy=dy[i]*10, width=1)
            ax3.arrow(x=x[-1], y=y[-1], dx=dx[-1]*15, dy=dy[-1]*15, width=2, color='yellow')
            if meth_error is not None:
                title = f"Last {len(x)} frames, landmarks. Diff between 3D-2D and 2D-2D: {meth_error:.2f}"
            else:
                title = f"Last {len(x)} frames, landmarks"
            ax3.set_title(title)

        plt.tight_layout()
        plt.pause(0.0001)

    # Close all windows we might have
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()