import cv2
import numpy as np

def getGaussianKernel(size, sigma):
    x = np.linspace(-(size - 1) / 2.0, (size-1)/2.0, size)
    gauss = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def getImageGradient(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    angle = np.arctan2(-sobely, sobelx) * 180/np.pi
    return magnitude, angle


def derotatePatch(img, loc, patch_size, orientation):
    # it can't be worse than a 45 degree rotation, so lets pad 
    # under this assumption. Then it will be enough for sure.
    padding = int(np.ceil(np.sqrt(2) * patch_size / 2))
    derotated_patch = np.zeros((patch_size,patch_size))
    padded_img = cv2.copyMakeBorder(img, padding, padding, 
            padding, padding, cv2.BORDER_CONSTANT, 0)
    ori_rad = orientation * np.pi/180
    
    # compute derotated patch  
    for px in range(patch_size):
        for py in range(patch_size):
          x_origin = px - patch_size/2
          y_origin = py - patch_size/2

          # rotate patch by angle ori
          x_rotated = np.cos(ori_rad) * x_origin - np.sin(ori_rad) * y_origin
          y_rotated = np.sin(ori_rad) * x_origin + np.cos(ori_rad) * y_origin

          # move coordinates to patch
          x_patch_rotated = loc[1] + x_rotated
          y_patch_rotated = loc[0] - y_rotated

          # sample image (using nearest neighbor sampling as opposed to more
          # accuracte bilinear sampling)
          y_img_padded = int(np.ceil(y_patch_rotated + padding))
          x_img_padded = int(np.ceil(x_patch_rotated + padding))
          derotated_patch[py, px] = padded_img[y_img_padded, x_img_padded]
    return derotated_patch


def computeDescriptors(blurred_images, keypoint_locations, rotation_invariant):
    num_octaves = len(blurred_images)
    assert(num_octaves == len(keypoint_locations))

    descriptors = []
    final_keypoint_locations = []

    # The factor of 1.5 is taken from Lowe's paper
    gauss_window = getGaussianKernel(16, 16 * 1.5)
    for oct_idx, (imgs, locs) in enumerate(zip(blurred_images, keypoint_locations)):

        # Only consider images that contain at least one keypoint, i.e.
        # where the image index appears at least once in locs[:, 2]
        for img_idx in np.unique(locs[:,2]):
            curr_img = imgs[:, :, img_idx]
            rows_img, cols_img = curr_img.shape[:2]
            Gmag, Gdir = getImageGradient(curr_img)

            # Select all keypoints that occur in the current image
            # and then discard the image information, e.g. keep only x, y position
            curr_loc = locs[locs[:,2] == img_idx, :2]

            num_keypoints = curr_loc.shape[0]
            curr_descriptors = np.zeros((num_keypoints, 128))
            is_valid_keypoint = np.zeros((num_keypoints,), dtype=bool)
            for idx_keypoint in range(num_keypoints):
                row, col = curr_loc[idx_keypoint, :]
                if row > 7 and col > 7 and row < rows_img-7 and col < cols_img-7:
                    is_valid_keypoint[idx_keypoint] = True

                    # get the local patches of gradient and direction
                    Gmag_loc = Gmag[row-8:row+8, col-8:col+8]
                    Gmag_loc_w = Gmag[row-8:row+8, col-8:col+8] * gauss_window
                    Gdir_loc = Gdir[row-8:row+8, col-8:col+8]

                    if rotation_invariant:
                        # compute dominant direction through looking at the most
                        # common orientation in the histogram, spaced at 10 deg
                        angle_edges = np.arange(-180, 181, 10) 
                        orient_hist, _ = np.histogram(Gdir_loc[:], 
                                bins = angle_edges, weights = Gmag_loc_w[:])
                        max_orient_idx = np.argmax(orient_hist)
                        Gdir_loc_principal = np.mean(angle_edges[max_orient_idx:max_orient_idx+1])
                        patch_derotated = derotatePatch(curr_img, [row, col], 
                            16, Gdir_loc_principal);
                        Gmag_derot, Gdir_derot = getImageGradient(patch_derotated)
                        Gmag_loc = Gmag_derot
                        Gmag_loc_w = Gmag_derot * gauss_window
                        Gdir_loc = Gdir_derot

                    N_tmp = 0
                    for ix in range(4):
                        for iy in range(4):
                            mag_4by4 = Gmag_loc_w[4*ix:4*(ix+1), 4*iy:4*(iy+1)]
                            dir_4by4 = Gdir_loc[4*ix:4*(ix+1), 4*iy:4*(iy+1)]
                            angles = np.arange(-180, 181, 45)
                            N_w, _ = np.histogram(dir_4by4[:], bins = angles, weights = mag_4by4[:])
                            curr_descriptors[idx_keypoint, N_tmp:N_tmp+8] = N_w
                            N_tmp += 8
            
            # Adapt keypoint location such that they correspond to the
            # originial image dimensions.
            curr_loc = curr_loc * 2**oct_idx
            # Only store valid keypoints.
            descriptors.append(curr_descriptors[is_valid_keypoint, :])
            final_keypoint_locations.append(curr_loc[is_valid_keypoint,:])

    descriptors = np.concatenate(descriptors)
    descriptors /= np.linalg.norm(descriptors, ord=2, axis=1, keepdims = True)
    final_keypoint_locations = np.concatenate(final_keypoint_locations)

    return descriptors, final_keypoint_locations 







