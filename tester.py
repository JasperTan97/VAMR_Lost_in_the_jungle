import cv2
import numpy as np
import os
import glob

DS_PATH = './data/parking/images/'
DS_GLOB = glob.glob(DS_PATH+'*.png')
BOOTSTRAP_FRAME = 1

# First, load in the images
img0 = None
img1 = None

for img_idx, img_path in enumerate(sorted(DS_GLOB)):
    # img = cv2.imread(os.path.join(DS_PATH, img_path))
    img = cv2.imread( img_path )
    cv2.imshow(f"Frame {img_idx} (file {img_path})", img)

    if img_idx == 0:
        img0 = img

    if img_idx == BOOTSTRAP_FRAME:
        img1 = img
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
img0_vis = cv2.drawKeypoints(img0_gray, kp0, img0)
img1_vis = cv2.drawKeypoints(img1_gray, kp1, img1)

cv2.imshow("KP0", img0_vis)
cv2.imshow("KP1", img1_vis)

# Naive matching code
SIFT_DIST_THRESH = 100.0     # Another tunable parameter
matches = []
for i, d0 in enumerate(des0):
    closest_idx = -1
    closest_dist = SIFT_DIST_THRESH
    for j, d1 in enumerate(des1):
        dist = np.linalg.norm(d0-d1)    
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = j

    matches.append((i, closest_idx, closest_dist))

print("Match indexes from image 0-> image 1")
print([x for x in matches if x[1]>=0])

# TODO verify this...

# Call 8-point Algorithm
pts0 = np.array([kp0[x[0]].pt for x in matches if x[1] >= 0])
pts1 = np.array([kp1[x[1]].pt for x in matches if x[1] >= 0])
F, _ = cv2.findFundamentalMat(pts0, pts1, cv2.RANSAC)

print("Fundemental Matrix:", F)

# TODO Get R, T out of fundemental matrix.

# Cleanup
cv2.waitKey(0)
cv2.destroyAllWindows()
