import cv2
import numpy as np
import scipy

def extractKeypoints(diff_of_gaussians, contrast_threshold):
    num_octaves = len(diff_of_gaussians)
    keypoint_locations = []
    
    for oct_idx, dog in enumerate(diff_of_gaussians):
        dog_max = scipy.ndimage.maximum_filter(dog, [3,3,3])
        is_keypoint = (dog == dog_max) & (dog >= contrast_threshold)
        is_keypoint[:, :, 0] = False
        is_keypoint[:, :, -1] = False
        keypoint_locations.append( np.array(is_keypoint.nonzero()).T )

    return keypoint_locations



