import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

DS_PATH = './data/parking/images/'
DS_GLOB = glob(DS_PATH+'*.png')
BOOTSTRAP_FRAME = 3

# First, load in the images (we just load from frame 0 and 3)
for img_idx, img_path in enumerate(sorted(DS_GLOB)):
    if img_idx == 0:
        img0 = cv2.imread( img_path )

    if img_idx == BOOTSTRAP_FRAME:
        img1 = cv2.imread( img_path )
        break

# Detect keypoints in images
img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Compute SIFT keypoints and descriptors
# kp is a cv2.keypoint object. Access pixel values with kp_.pt
# des is the descriptor, in this case a 128-long numpy float array.
kp0, des0 = sift.detectAndCompute(img0_gray, None)
kp1, des1 = sift.detectAndCompute(img1_gray, None)

# Match
# create BFMatcher object
bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
# Match descriptors
matches = bf.knnMatch(des0, des1, 2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.8*n.distance or n.distance < 0.8*m.distance:
        good_matches.append([m])

print("Good Matches:")
print([x[0].queryIdx for x in good_matches])

img3 = cv2.drawMatchesKnn(img0, kp0, img1, kp1, good_matches,
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.show()

# With these matches, we can now do RANSAC.

RANSAC_PROB_SUCCESS = 0.99      # Prob of success
RANSAC_OUTLIER_FRAC = 0.50      # Outlier ratio
RANSAC_NUM_MODEL_PARAMS = 8     # We are using 8-point algo, so 8 parameters
RANSAC_NUM_ITS = np.log(1-RANSAC_PROB_SUCCESS)/np.log(1-(1-RANSAC_OUTLIER_FRAC)**RANSAC_NUM_MODEL_PARAMS)

# Idea: Do RANSAC in parallel
for _ in range(RANSAC_NUM_ITS):
    # Select 8 points at random
    # Compute 8-point algo with these sampled points
    # For each datapoint:
        # Calculate the residual for each datapoint
        # Select datapoints in good_matches that support the hypothesis
# Select the set with the max number of inliers
# Calculate model parameteres again with all inliers
# Obtain R, T from essential matrix.

    pass

# cv2.waitKey(0)
cv2.destroyAllWindows()