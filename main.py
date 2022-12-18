import cv2
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict

from code.KLT_main import KLT

# Constants for tunable parameters
from constants import *

# For reading the dataset from file
from glob import glob

DATASET = 'parking'
if DATASET=='parking':
    DS_PATH = './data/parking/images/'
    K_PATH = './data/parking/K.txt'
elif DATASET=='kitti':
    raise NotImplementedError
elif DATASET=='malaga':
    raise NotImplementedError
else:
    raise ValueError

DS_GLOB = glob(DS_PATH+'*.png')

class VO_state:
    '''
    State that contains structs for
    - Point correspondences
        - P (2d coords (u,v) point correspondences)
        - X (3d coords (x,y,z) point positions)
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
        assert len(P) == len(X)

def featureDetection(image, method="SIFT") -> Tuple[List[Tuple[float,float]], List[np.ndarray]]:
    raise NotImplementedError

def triangulation(p0, p1, R, T):
    # Perform triangulation
    raise NotImplementedError

def get_R_T():
    raise NotImplementedError

def initialiseVO(I1, I0) -> VO_state:
    '''
    Bootstrapping

    Initializes
    '''
    # 1. Feature detection and descriptor generation
    P1, d1 = featureDetection(I1)
    P0, d0 = featureDetection(I0)

    # 2. Feature matching between I1, I0 features
    #    To obtain feature correspondences P0
    for i in range(len(P1)):
        # TODO brute-force matching (we have no priors)
        pass


    # 3. Normalized 5 or 8 point algo + RANSAC -> R, T between the two views
    # https://www.programcreek.com/python/example/89336/cv2.findFundamentalMat
    F, mask = cv2.findFundamentalMat(
            points1=P0,
            points2=P1,
            method=cv2.RANSAC
        )

    print(F, mask)  # TODO check if this returns the fundemental matrix

    # TODO get R, T out of F matrix
    R, T = get_R_T(F)

    # 4. Triangulation of all these landmarks
    X0 = triangulation(P0, P1, R, T)
    
    # 5. Bundle adjustment to refine R, T, X0
    # TODO figure it out


    return VO_state(P=P0, X=X0)

def processFrame(I1, I0, S0:VO_state) -> Tuple[VO_state, pose]:
    '''
    Continuous VO
    Inputs: Current image I1, previous state S0
    Outputs: Current state S1, current pose T1_WC

    State unpacks to P,X,C,F,T, see VO_state class
    '''
    P0, X0, C0, F0, T0 = S0 # unpack state i-1
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
    K = np.loadtxt(K_PATH)  # Get K (hardcoded to dataset)

    # Bootstrap
    for img_idx, img_path in enumerate(sorted(DS_GLOB)):
        if img_idx == 0:
            I0 = cv2.imread( img_path )

        if img_idx == BOOTSTRAP_FRAME:
            I1 = cv2.imread( img_path )
            break

    bootstrapped_state = initialiseVO(I1, I0)

    T0 = np.hstack((np.eye(3), np.zeros((3,1)))) # identity SE3 member for initial pose to signify world frame
    odom = [T0]

    # Continuous VO
    prev_state = bootstrapped_state # use bootstrapped state as first state

    for img_path in enumerate(sorted(DS_GLOB)):
        frame = cv2.imread(img_path)

        state, T_WC = processFrame(frame, prev_state) # continuous VO markov chain
        prev_state = state # set current state to previous state for next image
        odom.append(T_WC) # append current pose to odom list

        # ~ Press Q on keyboard to exit (for debugging)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    
    # Close all windows we might have
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()