from numpy import log
import math

STARTING_FRAME = 0
BOOTSTRAP_FRAME = 4


RANSAC_REPROJ_THRESHOLD = 0.1  # Reprojection error tolerated for RANSAC to consider it an inlier
                                # Value copied from MATLAB
RANSAC_PROB_SUCCESS = 0.99      # Probability of success of RANSAC
RANSAC_OUTLIER_FRAC = 0.50      # Outlier ratio
RANSAC_NUM_MODEL_PARAMS = 8     # We are using 8-point algo, so 8 parameters
RANSAC_NUM_ITS = int(
    log(1-RANSAC_PROB_SUCCESS)/log(1-(1-RANSAC_OUTLIER_FRAC)**RANSAC_NUM_MODEL_PARAMS)
)

########
# SIFT #
########
# User parameters
ROTATION_INVARIANT =True       # Enable rotation invariant SIFT

# sift parameters
CONTRAST_THRESHOLD = 0.04       # for feature matching
SIFT_SIGMA = 1.0                # sigma used for blurring
RESCALE_FACTOR = 1            # rescale images to make it faster
NUM_SCALES = 3                  # number of scales per octave
NUM_OCTAVES = 5                 # number of octaves

##########
# Harris #
##########
HARRIS_BLOCK_SIZE = 9       # Harris Neighborhood size.
HARRIS_K_SIZE = 3           # Aperture parameter for the Sobel operator.
HARRIS_K = 0.08             # Harris detector free parameter.
HARRIS_N_KPTS = 1000          # Number of kp responses to select
HARRIS_R = 9                # Non-max suppression radius

#######
# KLT #
#######
R_T = 31                        # scalar, radius of patch to track
KLT_N_ITER = 10                     # scalar, number of iterations
KLT_THRESHOLD = 1.0                 # scalar, bidirectional error threshold

##############
# PnP RANSAC #
##############

NUM_ITER_RANSAC = 1000
PIXEL_TOLERANCE = 10

#################
# Triangulation #
#################

PAR_THRESHOLD = math.cos(math.radians(0.25))
REPROJ_ERR = 5


