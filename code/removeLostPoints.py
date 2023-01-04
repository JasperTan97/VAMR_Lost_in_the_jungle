import numpy as np

def removeLostPoints(P0, X0, P1):
    """
    Function to remove the points lost in KLT tracking from previous state
    Input:
        P0: Previous image keypoint locations (2xM np.ndarray)
        P1: Current image keypoints tracked from P0 (2xM' np.ndarray)
        X0: Previous world points (3xM np.ndarray)
    Output:
        X1: Current World Points after removing lost points
    """
    return