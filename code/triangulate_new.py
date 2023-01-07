import numpy as np
import math
import ismember
from typing import Tuple
from code.triangulation import triangulation
from code.constants import *
import cv2

def TriangulateNew(P, X, C, F, T, T1_WC, K) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # if X.shape[1] > 400:
    #     return P, X, C, F, T
    X1 = np.copy(X)
    P1 = np.copy(P)
    rem_index = []
    # Identify unique blocks of T
    uniq, indices, inverse, counts = np.unique(T, axis=1, return_counts=True, return_index=True, return_inverse=True)
    n_unique = uniq.shape[1]
    # Triangulate for each T with current T1_WC
    for i in range(n_unique):
        Ti = T[:,indices[i]]
        projMat0 = np.hstack([Ti[:9].reshape([3,3]), Ti[9:].reshape(-1,1)])
        if (projMat0 == T1_WC).all():
            continue
        pts0 = F[:,indices[i]:indices[i]+counts[i]]
        projMat1 = np.copy(T1_WC)
        pts1 = np.vstack([C[indices[i]:indices[i]+counts[i],:].T, np.ones(counts[i])])
        X_new_CV, inFront = triangulation(pts0, pts1, K@projMat0, K@projMat1)
        # X_new_CV = cv2.triangulatePoints(K@projMat0, K@projMat1, pts0[:2,:], pts1[:2,:])
        # print("Check triangulation",np.mean(np.linalg.norm(X_new_CV - X_new_CV1, axis=0)))
        # X_new_cam = T_WC @ 
        
        # Check parallax for each set of triangulations
        n_pts = X_new_CV.shape[1]
        v0 = -(np.hstack(n_pts*[projMat0[:,-1].reshape(-1,1)]) - X_new_CV[:3,:])
        v1 = -(np.hstack(n_pts*[projMat1[:,-1].reshape(-1,1)]) - X_new_CV[:3,:])
        v0 = v0/np.linalg.norm(v0, axis=0)
        v1 = v1/np.linalg.norm(v1, axis=0)
        cosalpha = (v0*v1).sum(axis=0)
        # pts_bool = cosalpha < PAR_THRESHOLD
        pts_bool = np.logical_and(np.round(cosalpha,3) >= 0, cosalpha < PAR_THRESHOLD)

        check_pts1 = K@projMat1@X_new_CV
        check_pts1 /= check_pts1[2,:] 
        diff1 = check_pts1[:2,:]-pts1[:2,:]
        reprojError1 = np.linalg.norm(diff1, axis=0)

        check_pts2 = K@projMat0@X_new_CV
        check_pts2 /= check_pts2[2,:] 
        diff2 = check_pts2[:2,:]-pts0[:2,:]
        reprojError2 = np.linalg.norm(diff1, axis=0)

        meanReprojError = 0.5*(reprojError1+reprojError2)

        # meanReprojError = np.mean(np.linalg.norm(diff1, axis=0))

        pts_bool = np.logical_and(pts_bool, meanReprojError<4, inFront)

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
    
    return P1, X1, C1, F1, T1