import cv2
import numpy as np
import matplotlib.pyplot as plt

from code.compute_blurred_images import computeBlurredImages
from code.compute_descriptors import computeDescriptors 
from code.compute_difference_of_gaussians import computeDifferenceOfGaussians 
from code.compute_image_pyramid import computeImagePyramid 
from code.extract_keypoints import extractKeypoints

def SIFT(image, rotation_invariant, contrast_threshold, rescale_factor, sift_sigma, 
        num_scales, num_octaves):

    '''
    Input:
        image                   np.ndarray
        other constants         floats

    Output:
        keypoint_locations      np.ndarray, N x 2, where N is keypoint count of x, y coordinates
        keypoint_descriptors    np.ndarray, N x 128, where N is keypoint count
    '''
    image = cv2.normalize(
                cv2.resize( \
                        image, (0,0), fx = rescale_factor, fy = rescale_factor
                    ).astype('float'), \
                None, 0.0, 1.0, cv2.NORM_MINMAX)
    image_pyramid = computeImagePyramid(image, num_octaves)
    blurred_images = computeBlurredImages(image_pyramid, num_scales, sift_sigma)
    diff_of_gaussians = computeDifferenceOfGaussians(blurred_images)
    #print("Starting kpt locs with ", len(diff_of_gaussians), diff_of_gaussians[0].shape, contrast_threshold)
    tmp_keypoint_locs = extractKeypoints(diff_of_gaussians, contrast_threshold)
    #print("Starting Descriptors with ", len(blurred_images), blurred_images[0].shape, len(tmp_keypoint_locs), tmp_keypoint_locs[0].shape)
    desc, locs = computeDescriptors(blurred_images, tmp_keypoint_locs, rotation_invariant)
    #print("Returning")
    locs[:, [1, 0]] = locs[:, [0, 1]]
    return locs, desc