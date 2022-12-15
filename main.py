import cv2
import numpy as np

from dataclasses import dataclass
from typing import List, Tuple, Dict

# Constants for tunable parameters
from constants import *

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
    '''
    def __init__(self,
                P: List[np.ndarray]=[],
                X: List[np.ndarray]=[],
                C: List[np.ndarray]=None,
                F: List[np.ndarray]=None,
                T: List[np.ndarray]=None) -> None:
        assert len(P) == len(X)
        self.PX = []
        # TODO initalize P, X in an appropriate data structure
        # Perhaps init as a dict instead or with better indexing??
        for i in range(len(P)):
            self.PX.append( (P[i], X[i]) )


        if C is None or F is None or T is None:
            self.CFT = []
        assert len(C) == len(F)
        assert len(C) == len(T)
        # TODO initialize C, F, T in an appropriate data structure
        for i in range(len(C)):
            self.PX.append( (C[i], F[i], T[i]) )

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

def processFrame(I1, S0:VO_state) -> Tuple[VO_state, pose]:
    '''
    Continuous VO
    Inputs: Current image I1, previous state S0
    Outputs: Current state S1, current pose T1_WC

    State unpacks to P,X,C,F,T, see VO_state class
    '''
    P0, X0, C0, F0, T0 = S0 # unpack state i-1
    P1 = KLT(P0, I1) # tracks P0 features in I1
    X1 = removeLostPoints(P0, X0, P1) # Update X1 from P1 and X0
    R1, T1 = PnPRansac(P1, X1) # Get current pose with RANSAC
    T1_WC = np.hstack((R1, T1)) # Get transformation matrix in SE3 (without bottom row)
    C1 = KLT(C0, I1) # track C0 features in I1
    C1 = featureDectection(I1,C1,P1) # Add new features to keep C1 from shrinking
    F1 = firstObservationTracker(C0, F0, C1) # update F1 with C1
    T1 = cameraPoseMappedTo_c(C0, T0, C1) # update T1 with C1
    P1, X1, C1, F1, T1 = TriangulateProMaxPlusUltra(P1, X1, C1, F1, T1) # Add new points to pointcloud in P1, updates X1 accordingly, removes those points from C1, F1, and T1
    S1 = (P1, X1, C1, F1, T1) # repack state i
    
    return S1, T1_WC

def main() -> None:
    VIDEO_PATH = ''
    # Read from video stream
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Pick 2 keyframes to start before calling initialiseVO
    # Initialize VO from 2 views - I0 and I1 set manually
    BOOTSTRAP_FRAME = 3
    for i in range(BOOTSTRAP_FRAME):
        _, frame = cap.read()
        if i == 0:
            I0 = frame
        if i == BOOTSTRAP_FRAME-1:
            I1 = frame
    bootstrapped_state = initialiseVO(I1, I0)

    T0 = np.hstack((np.eye(3), np.zeros((3,1)))) # identity SE3 member for initial pose to signify world frame
    odom = [T0]

    # Continuous VO
    prev_state = bootstrapped_state # use bootstrapped state as first state
    while cap.isOpened():
        _, frame = cap.read() # get next image
        state, T_WC = processFrame(frame, prev_state) # continuous VO markov chain
        prev_state = state # set current state to previous state for next image
        odom.append(T_WC) # append current pose to odom list

        # ~ Press Q on keyboard to exit (for debugging)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything is done, release the video cap object.
    cap.release()

if __name__ == "__main__":
    main()