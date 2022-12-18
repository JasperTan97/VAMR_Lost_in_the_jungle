import numpy as np

from track_klt import trackKLT

def trackKLTRobustly(I_prev, I, keypoint, r_T, n_iter, threshold):
    """ 
    Input:
        I_R         np.ndarray reference image
        I           np.ndarray image to track points in
        x_T         1 x 2, point to track as [x y] = [col row]
        r_T         scalar, radius of patch to track
        n_iter      scalar, number of iterations
        threshold   scalar, bidirectional error threshold
    Output:
        delta       1 x 2, delta by which the keypoint has moved
        keep        boolean, true if the keypoint passes error test
    """ 
    W, _ =  trackKLT(I_prev, I, keypoint, r_T, n_iter)
    
    delta = W[:, -1]
    Winv, _ = trackKLT(I, I_prev, (keypoint.T + delta).T, r_T, n_iter)

    dkpinv = Winv[:, -1]
    keep = np.linalg.norm(delta + dkpinv) < threshold

    return delta, keep
