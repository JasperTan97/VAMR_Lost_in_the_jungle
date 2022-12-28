import numpy as np
from typing import Tuple

def vec_to_cross_mat(v:np.ndarray)->np.ndarray:
    '''Convert 3-vector to a 3x3 cross product matrix'''
    return np.array(
        [[    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],    0]]
    )

def normalise_2d_pts(pts:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    '''
    Normalize 2d homogenous points

    Function translates and normalises a set of 2D homogeneous points
    so that their centroid is at the origin and their mean distance from
    the origin is sqrt(2).

    Usage:   [pts_tilde, T] = normalise2dpts(pts)

    Argument:
    pts -  3xN array of 2D homogeneous coordinates

    Returns:
    pts_tilde -  3xN array of transformed 2D homogeneous coordinates.
    T         -  The 3x3 transformation matrix, pts_tilde = T*pts
    '''
    # Convert homogeneous coordinates to Euclidean coordinates (pixels)
    pts_ = pts/pts[2,:]

    # Centroid (Euclidean coordinates)
    mu = np.mean(pts_[:2,:], axis = 1)

    # Use RMS distance (another option is Average distance)
    pts_centered = (pts_[:2,:].T - mu).T

    # Option 1: RMS distance
    sigma = np.sqrt( np.mean( np.sum(pts_centered**2, axis = 0) ) )

    # Option 2: average distance
    # sigma = mean( sqrt(sum(pts_centered.^2)) );

    s = np.sqrt(2) / sigma
    T = np.array([
        [s, 0, -s * mu[0]],
        [0, s, -s * mu[1]],
        [0, 0, 1]])

    pts_tilde = T @ pts_

    return pts_tilde, T
