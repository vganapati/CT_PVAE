#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 21:57:15 2022

@author: vganapa1
"""

### Import python packages ###

import tomopy # always import before tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import xdesign as xd 
import matplotlib.pyplot as plt


tfd = tfp.distributions

### End of Imports ###

### Inputs ###

N_PIXEL = 128
SIZE_LOWER = 0.01
SIZE_UPPER = 0.2
GAP = 0
theta = np.linspace(0, np.pi, 180, endpoint=False) # projection angles
# reg = np.finfo(np.float32).eps.item() # regularization
poisson_noise_multiplier = 1e4 # poisson noise multiplier, 
                               # higher value means higher SNR
num_sparse_angles = 50 # number of angles to image per sample 
                       # (dose remains the same)
algorithm = 'gridrec' # 'sirt', 'gridrec'

### End of Inputs ###

if __name__ == '__main__':

    
    num_angles = len(theta)
    np.random.seed(0)
    
    # create ground truth
    phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, 
                      porosity=np.random.rand())
    phantom = xd.discrete_phantom(phantom, N_PIXEL)
    phantom = np.expand_dims(phantom,axis=0)
    
    plt.figure()
    plt.imshow(np.squeeze(phantom, axis=0),vmin=0,vmax=1)
    plt.colorbar()
    
    # create projection
    proj = tomopy.project(phantom, theta, center=None, emission=True, 
                          pad=True, sinogram_order=False)
    proj[proj<0] = 0 # remove small negative values
    
    plt.figure()
    plt.imshow(np.squeeze(proj, axis=1))
    
    
    # mask the projection
    sparse_angles = tf.random.shuffle(tf.range(num_angles))[:num_sparse_angles]
    mask = tf.expand_dims(tf.reduce_sum(tf.one_hot(sparse_angles,
                                                   num_angles,axis=0),
                                        axis=1),
                          axis=-1)
    mask = mask/num_sparse_angles
    
    proj_masked = proj*tf.expand_dims(mask, axis=-1)
    
    plt.figure()
    plt.imshow(np.squeeze(proj_masked, axis=1))
    
    # add Poisson-like noise to proj_masked
    proj_dist = tfd.Normal(loc = proj_masked,
                            scale = \
                                tf.sqrt(proj_masked/poisson_noise_multiplier))
            
    proj_sample = proj_dist.sample()  
            
    plt.figure()
    plt.imshow(np.squeeze(proj_sample, axis=1))
    
    
    # Tomopy: reconstruct unmasked, noiseless projection
        
    recon0 = tomopy.recon(proj, theta, center=None, 
                          algorithm=algorithm, sinogram_order=False)
    
    plt.figure()
    plt.imshow(np.squeeze(recon0), vmin=0,vmax=1)
    plt.colorbar()
    
    # Tomopy reconstruct noisy, masked
    used_angles = tf.squeeze(tf.cast(mask, tf.bool))
    recon1 = tomopy.recon(proj_sample[used_angles]*num_sparse_angles, theta[used_angles], 
                          center=None, algorithm=algorithm, 
                          sinogram_order=False)
        
    plt.figure()
    plt.imshow(np.squeeze(recon1),vmin=0,vmax=1)
    plt.colorbar()
    
    # Exercises:
    # Uniform angle spacing with increasing number of angles
    # Limited view tomography: a dense cluster of angles used
