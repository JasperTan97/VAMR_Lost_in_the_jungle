import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from code.constants import RANSAC_REPROJ_THRESHOLD, RANSAC_PROB_SUCCESS, RANSAC_NUM_ITS

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
# Ref: https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
# des is the descriptor, in this case a 128-long numpy float array.
kp0, des0 = sift.detectAndCompute(img0_gray, None)
kp1, des1 = sift.detectAndCompute(img1_gray, None)

# print(len(kp0), len(kp1))
print([kp.pt for kp in kp0[:10]])

# Match
# create BFMatcher object
bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
# Match descriptors
matches = bf.knnMatch(des0, des1, k=2)
# knnMatch returns tuple of 2 nearest matches
# print(len(matches))
# print([f"{m.trainIdx}->{m.queryIdx}:{m.distance}; {n.trainIdx}->{n.queryIdx}:{n.distance}" for m, n in matches[:10]])
# Ref: https://docs.opencv.org/4.x/d4/de0/classcv_1_1DMatch.html
# the query index is from the 1st arg (in this case des0)
# the train index is from the 2nd arg (in this case des1)

# Apply ratio test
good_matches = []
# m is the best match, n is the second-best.
# Access the distance between matches using <>.distance
for m, n in matches:
    if m.distance < 0.8*n.distance or n.distance < 0.8*m.distance:
        good_matches.append([m])    # Visualisation requires list of match objects

# print("Good Matches:")
# print([f"{x[0].trainIdx}->{x[0].queryIdx}" for x in good_matches])
pts0 = np.array([kp0[x[0].queryIdx].pt for x in good_matches])
pts1 = np.array([kp1[x[0].trainIdx].pt for x in good_matches])
print(pts0.shape)

# Visualise
img3 = cv2.drawMatchesKnn(img0, kp0, img1, kp1, good_matches,
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.show()


'''
pts0 and pts1 are Nx2 Numpy arrays containing the pixel coords of the matches.
'''
F, _ = cv2.findFundamentalMat(
    pts0, 
    pts1,
    method=cv2.FM_RANSAC, 
    ransacReprojThreshold=RANSAC_REPROJ_THRESHOLD,
    confidence=RANSAC_PROB_SUCCESS,
    maxIters=RANSAC_NUM_ITS
)

# Extract out correspondences

# cv2.waitKey(0)
cv2.destroyAllWindows()