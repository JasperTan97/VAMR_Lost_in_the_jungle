import cv2
import numpy as np

def computeBlurredImages(image_pyramid, num_scales, sift_sigma):
    num_octaves = len(image_pyramid)
    imgs_per_oct = num_scales + 3
    
    blurred_images = []
    for oct_idx, img in enumerate(image_pyramid):
        octave_stack = np.zeros(np.r_[img.shape, imgs_per_oct])
        for stack_idx in range(imgs_per_oct):
            gauss_blur_sigma = sift_sigma * 2**((stack_idx-1)/num_scales)
            filter_size = int(2*np.ceil(2*gauss_blur_sigma)+1.0)
            octave_stack[:, :, stack_idx] = \
                    cv2.GaussianBlur(img, (filter_size,filter_size), gauss_blur_sigma)
        blurred_images.append(octave_stack)
    
    return blurred_images
        
