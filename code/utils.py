import numpy as np


def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M



def distPoint2EpipolarLine(F, p1, p2):
    """ Compute the point-to-epipolar-line distance

       Input:
       - F np.ndarray(3,3): Fundamental matrix
       - p1 np.ndarray(3,N): homogeneous coords of the observed points in image 1
       - p2 np.ndarray(3,N): homogeneous coords of the observed points in image 2

       Output:
       - cost: sum of squared distance from points to epipolar lines
               normalized by the number of point coordinates
    """

    N = p1.shape[1]

    homog_points = np.c_[p1, p2]
    epi_lines = np.c_[F.T @ p2, F @ p1]

    denom = epi_lines[0,:]**2 + epi_lines[1,:]**2
    cost = np.sqrt( np.sum( np.sum( epi_lines * homog_points, axis = 0)**2 / denom) / N)

    return cost
