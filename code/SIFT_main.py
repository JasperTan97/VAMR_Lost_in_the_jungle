import cv2
import numpy as np
import matplotlib.pyplot as plt

from compute_blurred_images import computeBlurredImages
from compute_descriptors import computeDescriptors 
from compute_difference_of_gaussians import computeDifferenceOfGaussians 
from compute_image_pyramid import computeImagePyramid 
from extract_keypoints import extractKeypoints

def main(image, rotation_invariant, contrast_threshold, sift_sigma, 
        num_scales, num_octaves):

    '''
    Input:
        image                   np.ndarray
        other constants         floats

    Output:
        keypoint_locations      np.ndarray, N x 2, where N is keypoint count of x, y coordinates
        keypoint_descriptors    np.ndarray, N x 128, where N is keypoint count
    '''

    image_pyramid = computeImagePyramid(image, num_octaves)
    blurred_images = computeBlurredImages(image_pyramid, num_scales, sift_sigma)
    diff_of_gaussians = computeDifferenceOfGaussians(blurred_images)
    tmp_keypoint_locs = extractKeypoints(diff_of_gaussians, contrast_threshold)
    desc, locs = computeDescriptors(blurred_images, tmp_keypoint_locs, rotation_invariant)

    return locs, desc

