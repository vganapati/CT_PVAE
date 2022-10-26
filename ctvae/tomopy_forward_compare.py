#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:00:44 2022

@author: vganapa1

Adapted from Minh and Rey's E56 Final Project
"""
 
import sys
import tomopy # always import tomopy before tensorflow
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import xdesign as xd
import matplotlib.pyplot as plt
import time
from .forward_functions import project_tf_fast, project_tf_low_mem

tfd = tfp.distributions

def main():
    theta = np.linspace(0, np.pi, 100, endpoint=False)
    num_angles = len(theta)
    n_pixel = 128
    size_lower = 0.01
    size_upper = 0.2
    gap=0
    slice_ind = 1 # slice to visualize
    
    np.random.seed(0)
    phantom_0 = xd.Foam(size_range=[size_upper, size_lower], gap=gap,
                     porosity=np.random.rand())
    phantom_0 = xd.discrete_phantom(phantom_0, n_pixel)
    
    phantom_1 = xd.Foam(size_range=[size_upper, size_lower], gap=gap,
                     porosity=np.random.rand())
    phantom_1 = xd.discrete_phantom(phantom_1, n_pixel)
    
    phantom = tf.stack((phantom_0, phantom_1), axis=-1)
    
    plt.figure()
    plt.imshow(phantom[:,:,slice_ind], 'gray')
    plt.colorbar()
    plt.title('Object')
    plt.show()
    
    
    
    start_time = time.time()
    proj_fast = project_tf_fast(phantom,theta,pad=True)
    fast_time = time.time()-start_time
    
    start_time = time.time()
    proj_low_mem = project_tf_low_mem(phantom,theta, pad=True)
    mem_time = time.time()-start_time
    
    start_time = time.time()
    proj = tomopy.project(tf.transpose(phantom, perm=[2,0,1]), theta, center=None, 
                          emission=True, pad=True, sinogram_order=False)
    proj = tf.transpose(proj, perm=[0,2,1])
    tomopy_time = time.time()-start_time
    
    print('fast time: ',round(fast_time,4),'seconds')
    print('low memory time: ',round(mem_time,4),'seconds')
    print('tomopy time: ',round(tomopy_time,4),'seconds')
    
    plt.figure() 
    plt.title('our sinogram')
    # periodic boundary conditions at top and bottom
    plt.imshow(proj_fast[:,:,slice_ind], cmap='gray')
    plt.colorbar()
    plt.xlabel('projection line')
    plt.ylabel('angle')
    plt.show()
    
    plt.figure() 
    plt.title('tomopy sinogram')
    # periodic boundary conditions at top and bottom
    plt.imshow(proj[:,:,slice_ind], cmap='gray')
    plt.colorbar()
    plt.xlabel('projection line')
    plt.ylabel('angle')
    plt.show()
    
    print(tf.transpose(proj_fast,perm=(0,2,1)).shape)
    print(proj.shape)
    
    recon0 = tomopy.recon(tf.transpose(proj_fast,perm=(0,2,1)), theta, center=None, algorithm='sirt',
                         sinogram_order=False)
    recon1 = tomopy.recon(tf.transpose(proj,perm=(0,2,1)), theta, center=None, algorithm='sirt',
                         sinogram_order=False)
    print(recon0.shape)
    print(recon1.shape)
    
    plt.figure()
    plt.title('Reconstruction from tomopy projection')
    plt.imshow(recon1[slice_ind], cmap='gray')
    plt.colorbar()
    
    plt.figure()
    plt.title('Reconstruction from our fast projection')
    plt.imshow(recon0[slice_ind], cmap='gray')
    plt.colorbar()
    
    plt.figure()
    plt.title('Reconstruction Difference')
    plt.imshow(recon0[slice_ind]-recon1[slice_ind], cmap='gray')
    plt.colorbar()

if __name__ == "__main__":
    main()