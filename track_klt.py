import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

from get_sim_warp import getSimWarp
from get_warped_patch import getWarpedPatch

def trackKLT(I_R, I, x_T, r_T, n_iter):
    """ 
    Input:
        I_R     np.ndarray reference image
        I       np.ndarray image to track points in
        x_T     1 x 2, point to track as [x y] = [col row]
        r_T     scalar, radius of patch to track
        n_iter  scalar, number of iterations
    Output:
        estimated warp
        history of parameter estimates ( 6 x (n_iter + 1) ) 
            including the initial identity estimate
    """ 
    
    p_hist = np.zeros((6, n_iter + 1))
    W = getSimWarp(0, 0, 0, 1)
    p_hist[:, 0] = np.reshape(W.T, (1, -1))

    # Suffix T indicates image evaluated for patch T
    I_RT = getWarpedPatch(I_R, W, x_T, r_T)

    # The template vector is fixed for all time
    i_R = np.reshape(I_RT.T, (-1, ))

    # The x and y coordinates of the patch are fixed as well
    xs = np.arange(-r_T, r_T+1)
    ys = np.arange(-r_T, r_T+1)
    n = xs.shape[0]
    xy1 = np.c_[
            np.kron(xs, np.ones((1, n))).T,
            np.kron(np.ones((1, n)), ys).T,
            np.ones((n*n, 1)) ]
    dwdx = np.kron(xy1, np.eye(2))
    
    do_plot = False
    fig = None

    if do_plot:
        fig = plt.figure()

    ct = 0
    for it in range(n_iter):
        # for the convolution below to remain valid, get a bit larger patch
        big_IWT = getWarpedPatch(I, W, x_T, r_T+1)
        IWT = big_IWT[1:-1, 1:-1]
        i = np.reshape(IWT.T, (-1,))

        # computing di/dp
        IWTx = convolve2d(np.array([1, 0, -1], ndmin=2), big_IWT[1:-1, :], 'valid')
        IWTy = convolve2d(np.array([[1], [0], [-1]], ndmin=2), big_IWT[:, 1:-1], 'valid')
        didw = np.c_[np.reshape(IWTx.T, (-1, 1)), np.reshape(IWTy.T, (-1, 1))]
        didp = np.zeros((n*n, 6))
        
        # Efficient
        didp = (didw[:,0] * dwdx[::2,:].T + didw[:,1] * dwdx[1::2,:].T).T
        # THIS DOES THE SAME AS THE VECTORIZED IMPLEMENTATION ABOVE
        #  for px_i in range(n*n):
            #  didp[px_i, :] = didw[px_i, :] @ dwdx[2*px_i:2*(px_i+1),:]

        # Hessian
        H = didp.T @ didp

        # update step
        delta_p = np.linalg.pinv(H) @ didp.T @ (i_R - i)
        W += np.reshape(delta_p, (3, 2)).T
        
        if do_plot:
            plt.clf()
            ax = fig.add_subplot(311)
            ax.imshow(np.c_[IWT, I_RT, I_RT-IWT])
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('I(W(T)), I_R(T) and their difference')
            
            ax = fig.add_subplot(312)
            ax.imshow(np.c_[IWTx, IWTy])
            ax.set_title("Warped gradients")

            ax = fig.add_subplot(313)
            descentcat = np.zeros((n, 6*n))
            for j in range(6):
                descentcat[:, j*n:(j+1)*n] = np.reshape(didp[:,j], (n,n))
            ax.imshow(descentcat)
            ax.set_title('steepest descent images')
            plt.pause(0.1)

        p_hist[:, it + 1] = np.reshape(W, (1, -1))

        if np.linalg.norm(delta_p) < 1e-3:
            p_hist = p_hist[:, :it+1]
            return W, p_hist
    
    return W, p_hist

        
    
    

        

