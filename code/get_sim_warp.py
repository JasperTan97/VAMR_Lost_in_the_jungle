import numpy as np

def getSimWarp(dx, dy, alpha, scale):
    """
    Input
        dx      scalar, translation x
        dy      scalar, translation y
        alpha   scalar, rotation angle _in degrees_
        scale   scalar, scale 
    
    Output
        W       2 x 3 np.array 
    """

    alpha_rad = alpha * np.pi / 180
    c = np.cos(alpha_rad)
    s = np.sin(alpha_rad)
    R = np.array([[c, -s], [s, c]])
    W = scale * np.c_[R, np.array([[dx],[dy]])]

    return W
