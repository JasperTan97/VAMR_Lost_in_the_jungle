import numpy as np
import math
import ismember
from typing import Tuple
from code.triangulation import triangulation
from code.constants import *

def TriangulateNew(P, X, C, F, T, T1_WC) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[1] > 400:
        return P, X, C, F, T
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
        X_new = triangulation(pts0, pts1, projMat0, projMat1)
        # Check parallax for each set of triangulations
        n_pts = X_new.shape[1]
        v0 = np.hstack(n_pts*[projMat0[:,-1].reshape(-1,1)]) - X_new[:3,:]
        v1 = np.hstack(n_pts*[projMat1[:,-1].reshape(-1,1)]) - X_new[:3,:]
        v0 = v0/np.linalg.norm(v0, axis=0)
        v1 = v1/np.linalg.norm(v1, axis=0)
        cosalpha = (v0*v1).sum(axis=0)
        pts_bool = np.logical_and(cosalpha > 0, cosalpha < PAR_THRESHOLD)
        # Add required points to X1 and P1 using C1
        pts_add = X_new[:,pts_bool]
        X = np.hstack([X, pts_add])
        P = np.vstack([P, pts1[:2,pts_bool].T])
        # Remove added points from C, F, T
        rem_index = rem_index + np.arange(indices[i], indices[i]+counts[i])[pts_bool].tolist()
    C = np.delete(C, rem_index, axis=0)
    F = np.delete(F, rem_index, axis=1)
    T = np.delete(T, rem_index, axis=1)
    
    return P, X, C, F, T