import cv2
import numpy as np

from typing import List, Tuple, Dict

from code.KLT_main import KLT
from code.get_relative_pose import get_relative_pose
from code.SIFT_main import SIFT
from code.linear_triangulation import linearTriangulation

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
    raise NotImplementedError
elif DATASET=='malaga':
    raise NotImplementedError
else:
    raise ValueError

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

        assert P.shape[-1] == X.shape[-1], f"P (shape {P.shape}) and X (shape {X.shape}) have diff lengths"
        assert P.shape[0] == 3, "P are the homogenous pixel correspondences, it should have structure (u,v,1)"
        assert X.shape[0] == 4, "X are the homogenous 3d point correspondences, it should have structure (x,y,z,1)"
        self.P = P
        self.X = X

        if C is None or F is None or T is None:
            self.C = []
            self.F = []
            self.T = []
        else:
            assert len(C) == len(F)
            assert len(C) == len(T)
            # TODO initialize C, F, T in an appropriate data structure
            self.C = C
            self.F = F
            self.T = T

def featureDetection(image, method="SIFT") -> Tuple[np.ndarray, np.ndarray]: # returns locations, descriptions
    if method == "SIFT":
        return SIFT(image, ROTATION_INVARIANT, CONTRAST_THRESHOLD, SIFT_SIGMA, NUM_SCALES, NUM_OCTAVES)

# def triangulation(p0: np.ndarray, p1: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
#     # Perform triangulation
#     # first make each p in p0 and p1 a 3x1 vector, currently a 2x1
#     p0 = np.concatenate((p0, np.ones((p0.shape[0],1))), axis=1)
#     p1 = np.concatenate((p1, np.ones((p0.shape[0],1))), axis=1)
#     # make M1 : I(3x3) + 0(3x1)
#     M0 = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)
#     # make M2 : R + T (concatenated)
#     M1 = np.concatenate((R, T), axis=1)
#     # run linear triangulation
#     X_4byN = linearTriangulation(p0.T, p1.T, M0, M1)
#     return np.delete(X_4byN, -1, 0).T # returns Nx3 array of world coordinates

def get_R_T():
    raise NotImplementedError

def initialiseVO(I1, I0) -> VO_state:
    '''
    Bootstrapping

    Initializes
    '''
    # 1. Feature detection and descriptor generation
    # TODO @Abhiram replace this without using cv2.sift code
    img0_gray = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Compute SIFT keypoints and descriptors
    # kp is a cv2.keypoint object. Access pixel values with kp_.pt
    # Ref: https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
    # des is the descriptor, in this case a 128-long numpy float array.
    kp0, des0 = sift.detectAndCompute(img0_gray, None)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)

    # P1, d1 = featureDetection(I1)
    # P0, d0 = featureDetection(I0)

    # 2. Feature matching between I1, I0 features
    #    To obtain feature correspondences P0

    # create BFMatcher object
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    # Match descriptors
    matches = bf.knnMatch(des0, des1, k=2)
    # knnMatch returns tuple of 2 nearest matches
    # print(len(matches))
    # print([f"{m.trainIdx}->{m.queryIdx}:{m.distance}; {n.trainIdx}->{n.queryIdx}:{n.distance}" for m, n in matches[:10]])
    # Ref: https://docs.opencv.org/4.x/d4/de0/classcv_1_1DMatch.html
    # the query index is from the 1st arg (in this case des0)
    # the train index is from the 2nd arg (in this case des1)

    # Apply ratio test
    good_matches = []
    # m is the best match, n is the second-best.
    # Access the distance between matches using <>.distance
    for m, n in matches:
        if m.distance < 0.8*n.distance or n.distance < 0.8*m.distance:
            good_matches.append([m])    # Visualisation requires list of match objects

    # 3. Get Fundemental matrix
    pts0 = np.array([kp0[x[0].queryIdx].pt for x in good_matches])
    pts1 = np.array([kp1[x[0].trainIdx].pt for x in good_matches])

    # pts0 and pts1 are Nx2 Numpy arrays containing the pixel coords of the matches.
    F, _ = cv2.findFundamentalMat(
        pts0, 
        pts1,
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
        confidence=RANSAC_PROB_SUCCESS,
        maxIters=RANSAC_NUM_ITS
    )

    # Essential Matrix
    E = np.linalg.inv(K.T) @ F @ np.linalg.inv(K)

    # Convert points format for OpenCV into 3xN homogenous pixel coordinates
    points0 = np.vstack([pts0.T, np.ones((1, pts0.shape[0]))])
    points1 = np.vstack([pts1.T, np.ones((1, pts1.shape[0]))])

    # Get R, T, X out of E matrix
    # X is a byproduct of disambiguating the possibilities in R, T
    _, _, X0 = get_relative_pose(
        points0,
        points1,
        E, K
    )
    # print(f"Points: {points0.shape}, 3D: {X0.shape}")

    # triangulation already done in 3. Removed 4. triangulation

    # 5. Bundle adjustment to refine R, T, X0
    # TODO figure this out

    return VO_state(P=points0, X=X0)

def processFrame(I1, I0, S0: VO_state) -> Tuple[VO_state, Pose]:
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

    P1 = KLT(P0, I1, I0) # tracks P0 features in I1
    X1 = removeLostPoints(P0, X0, P1) # Update X1 from P1 and X0
    R1, T1 = PnPRansac(P1, X1) # Get current pose with RANSAC
    T1_WC = np.hstack((R1, T1)) # Get transformation matrix in SE3 (without bottom row)
    C1 = KLT(C0, I1, I0) # track C0 features in I1
    C1 = featureDectection(I1,C1,P1) # Add new features to keep C1 from shrinking
    F1 = firstObservationTracker(C0, F0, C1) # update F1 with C1
    T1 = cameraPoseMappedTo_c(C0, T0, C1) # update T1 with C1
    P1, X1, C1, F1, T1 = TriangulateProMaxPlusUltra(P1, X1, C1, F1, T1) # Add new points to pointcloud in P1, updates X1 accordingly, removes those points from C1, F1, and T1
    S1 = (P1, X1, C1, F1, T1) # repack state i
    
    return S1, T1_WC

def main() -> None:
    # Bootstrap
    for img_idx, img_path in enumerate(DS_GLOB):
        if img_idx == 0:
            I0 = cv2.imread( img_path )

        if img_idx == BOOTSTRAP_FRAME:
            I1 = cv2.imread( img_path )
            break

    bootstrapped_state = initialiseVO(I1, I0)
    print(bootstrapped_state.P)
    print(bootstrapped_state.X)

    T0 = np.hstack((np.eye(3), np.zeros((3,1)))) # identity SE3 member for initial pose to signify world frame
    odom = [T0]

    # Continuous VO
    prev_state = bootstrapped_state # use bootstrapped state as first state
    prev_frame = I0

    for img_path in DS_GLOB[1:]:
        frame = cv2.imread(img_path)

        state, T_WC = processFrame(frame, prev_frame, prev_state) # continuous VO markov chain

        prev_state = state # set current state to previous state for next image
        prev_frame = frame

        odom.append(T_WC) # append current pose to odom list

        # ~ Press Q on keyboard to exit (for debugging)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    
    # Close all windows we might have
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()