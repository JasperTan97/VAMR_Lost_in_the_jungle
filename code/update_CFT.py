import numpy as np
import ismember
from typing import Tuple
from KLT_main import KLT
from SIFT_main import SIFT
from constants import *
from main import featureDectection # TODO: Change name of main

def update_CFT(prev_state:VO_state, prevImg, currImg, posePoints):
    # Detect features in the current image
    C_new, _ = SIFT(image, ROTATION_INVARIANT, CONTRAST_THRESHOLD, SIFT_SIGMA, NUM_SCALES, NUM_OCTAVES)
    # Check and remove the features which are currently tracked for pose detection
    L, Locs = ismember(C_new, posePoints, "rows")
    C_new = C_new[~L, :]
    # Track candidate keypoints from previous state
    keypoints_candidates = KLT(prev_state.C, currImg, prevImg)
    
    # Update C, F, T values for the candidate keypoints which were previously tracked
    return