import numpy as np
import time

def getWarpedPatch(I, W, x_T, r_T):
    """
    Input
        I   np.ndarray image
        W   2 x 3 np.ndarray
        x_T 1 x 2 np.ndarray
        r_T scalar

    Output
        (2 * r_T + 1) x (2 * r_T + 1) np.ndarray image patch
    """
    t0 = time.time()
    patch = np.zeros((2 * r_T + 1, 2 * r_T + 1))
    interpolate = True

    # minimal and maximal coordinates that are valid in (x,y) style
    min_coords = np.array([0, 0])
    max_coords = np.array(I.shape[::-1]) - 1

    # to achieve a speed comparable to MATLAB, we need to spend some 
    # effort on this function
    xm, ym = np.meshgrid(np.arange(-r_T, r_T+1), np.arange(-r_T, r_T+1))
    xm = np.reshape(xm, (1,-1))
    ym = np.reshape(ym, (1,-1))
    pre_warp = np.r_[xm, ym, np.ones_like(xm)]
    warped = (x_T[:,np.newaxis] + (W @ pre_warp)).T
    mask = np.logical_or.reduce(np.c_[
        warped[:,0] >= max_coords[0],
        warped[:,0] <= min_coords[0],
        warped[:,1] >= max_coords[1],
        warped[:,1] <= min_coords[1]], axis=1)
    warped[mask,:] = 0

    if interpolate:
        # perform bilinear interpolation
        floors = np.floor(warped).astype('int')
        weights = warped - floors
        image_int = ((1-weights[:,0]) * I[floors[:,1], floors[:,0]] \
                    + weights[:,0] * I[floors[:,1], floors[:,0]+1]) \
                    * (1 - weights[:,1]) \
                    + ((1-weights[:,0]) * I[floors[:,1]+1, floors[:,0]] \
                    + weights[:,0] * I[floors[:,1]+1, floors[:,0]+1]) \
                    * weights[:,1]
        image_int[mask] = 0
        patch = np.reshape(image_int, (2*r_T+1, 2*r_T+1))
    else:
        warped_int = warped.astype('int')
        image_int = I[warped_int[:,1], warped_int[:,0]]
        image_int[mask] = 0
        patch = np.reshape(image_int, (2*r_T+1, 2*r_T+1))
        
    # THIS CODE IS IDENTICAL TO THE VECTORIZED IMPLEMENTATION ABOVE
    # BUT ABOUT 250x FASTER
    #  for x in range(-r_T, r_T+1):
        #  for y in range(-r_T, r_T+1):
            #  pre_warp = np.array([x, y, 1])
            #  warped = x_T + pre_warp @ W.T
            #  if np.all(warped < max_coords) and np.all(warped > min_coords):
                #  if interpolate:
                    #  # perform bilinear interpolation
                    #  floors = np.floor(warped).astype('int')
                    #  weights = warped - np.floor(warped)
                    #  a = weights[0]
                    #  b = weights[1]
                    #  intensity = (1-b) * (
                        #  (1-a) * I[floors[1], floors[0]] +
                        #  a * I[floors[1], floors[0]+1]) \
                        #  + b * (
                        #  (1-a) * I[floors[1]+1, floors[0]] +
                        #  a * I[floors[1]+1, floors[0]+1]);
                    #  patch[y + r_T, x + r_T] = intensity
                #  else:
                    #  patch[y + r_T, x + r_T] = I[int(warped[1]), int(warped[0])]

    return patch

