import numpy as np
import math
#import ismember
from typing import Tuple
from code.triangulation import triangulation
from code.constants import *
import cv2

def TriangulateNew(P, X, C, F, T, T1_CW, K) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # if X.shape[1] > 400:
    #     return P, X, C, F, T
    X1 = np.copy(X)
    P1 = np.copy(P)
    rem_index = []
    # Identify unique blocks of T (in the same camera pose)
    uniq, indices, inverse, counts = np.unique(T, axis=1, return_counts=True, return_index=True, return_inverse=True)
    n_unique = uniq.shape[1]
    # Triangulate for each T with current T1_CW
    for i in range(n_unique):
        Ti = T[:,indices[i]]
        projMat0 = np.hstack([Ti[:9].reshape([3,3]), Ti[9:].reshape(-1,1)])
        # No need to attempt to triangulate if the keypoint is first detected in the current frame
        if np.allclose(projMat0, T1_CW):
            continue

        # Distance to current frame (assume increasing)
        # If this distance is too large, then we break, as the values can no longer be trusted.
        # dist_to_curr_frame = np.linalg.norm((T1_CW[:,3]-Ti[9:]))
        # if dist_to_curr_frame > 2.0:
        #     break

        # Y-rotation to current frame
        ang_between_frames = np.arctan2(Ti[2],Ti[0])

        # Find all pixel coordinates
        pts0 = F[:,indices[i]:indices[i]+counts[i]]
        projMat1 = np.copy(T1_CW)
        pts1 = np.vstack([C[indices[i]:indices[i]+counts[i],:].T, np.ones(counts[i])])
        X_new_CV = triangulation(pts0, pts1, K@projMat0, K@projMat1)
        inFront = (T1_CW @ X_new_CV)[2,:]>0
        # X_new_CV = cv2.triangulatePoints(K@projMat0, K@projMat1, pts0[:2,:], pts1[:2,:])
        # print("Check triangulation",np.mean(np.linalg.norm(X_new_CV - X_new_CV1, axis=0)))
        # X_new_cam = T_CW @ 
        n_pts = X_new_CV.shape[1]
        #print(f"1. Fraction infront:{(inFront.shape[0]/X_new_CV.shape[1]):.2f}, #Pts:{n_pts}, Dist:{dist_to_curr_frame:.3f}, Ang:{np.degrees(ang_between_frames):.2f}°")
        # Check parallax for each set of triangulations
        v0 = -(np.hstack(n_pts*[projMat0[:,-1].reshape(-1,1)]) - X_new_CV[:3,:])
        v1 = -(np.hstack(n_pts*[projMat1[:,-1].reshape(-1,1)]) - X_new_CV[:3,:])
        v0 = v0/np.linalg.norm(v0, axis=0)
        v1 = v1/np.linalg.norm(v1, axis=0)
        cosalpha = (v0*v1).sum(axis=0)
        # pts_bool = cosalpha < PAR_THRESHOLD
        pts_bool = np.logical_and(np.round(cosalpha,3) >= 0, cosalpha < PAR_THRESHOLD)

        # Reprojection error of points in the current camera pose
        check_pts1 = K@projMat1@X_new_CV
        check_pts1 /= check_pts1[2,:]
        diff1 = check_pts1[:2,:]-pts1[:2,:]
        reprojError1 = np.linalg.norm(diff1, axis=0)

        # Reprojection error of points in the past camera pose
        check_pts2 = K@projMat0@X_new_CV
        check_pts2 /= check_pts2[2,:]
        diff2 = check_pts2[:2,:]-pts0[:2,:]
        reprojError2 = np.linalg.norm(diff2, axis=0)

        meanReprojError = 0.5*(reprojError1+reprojError2)

        # meanReprojError = np.mean(np.linalg.norm(diff1, axis=0))

        pts_bool = np.logical_and(pts_bool, meanReprojError<REPROJ_ERR, inFront)

        # Add required points to X1 and P1 using C1
        pts_add = X_new_CV[:,pts_bool]
        X1 = np.hstack([X1, pts_add])
        P1 = np.vstack([P1, pts1[:2,pts_bool].T])

        # print(reprojError)
        # Remove added points from C, F, T
        rem_index = rem_index + np.arange(indices[i], indices[i]+counts[i])[pts_bool].tolist()

    C1 = np.delete(C, rem_index, axis=0)
    F1 = np.delete(F, rem_index, axis=1)
    T1 = np.delete(T, rem_index, axis=1)
    # print(-C1.shape[0]+C.shape[0])
    # X_cam = T1_CW @ X1

    # n_front = np.sum(X_cam[2,:] > 0)
    #print("1. Fraction of points infront", inFront.shape[1]/X1.shape[1], X1.shape[1])

    return P1, X1, C1, F1, T1