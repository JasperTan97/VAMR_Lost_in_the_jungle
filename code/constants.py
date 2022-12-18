from numpy import log

BOOTSTRAP_FRAME = 3

RANSAC_REPROJ_THRESHOLD = 0.01  # Reprojection error tolerated for RANSAC to consider it an inlier
                                # Value copied from MATLAB
RANSAC_PROB_SUCCESS = 0.99      # Probability of success of RANSAC
RANSAC_OUTLIER_FRAC = 0.50      # Outlier ratio
RANSAC_NUM_MODEL_PARAMS = 8     # We are using 8-point algo, so 8 parameters
RANSAC_NUM_ITS = int(
    log(1-RANSAC_PROB_SUCCESS)/log(1-(1-RANSAC_OUTLIER_FRAC)**RANSAC_NUM_MODEL_PARAMS)
)