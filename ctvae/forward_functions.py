#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:58:57 2022

@author: vganapa1

Adapted from Minh and Rey's E56 Final Project
"""
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import numpy as np


tfd = tfp.distributions

def pad_phantom(phantom, dim = 3,
                integrate_vae = False):
    
    if integrate_vae: # first dimension is the batch size
        img_size_x = phantom.shape[1]
        img_size_y = phantom.shape[2]
    else:
        img_size_x = phantom.shape[0]
        img_size_y = phantom.shape[1]
    
    # img_size_z = phantom.shape[2]
    num_proj_pix = tf.sqrt(tf.cast(img_size_x**2 + img_size_y**2, tf.float64)) + 2
    num_proj_pix = tf.cast(tf.math.ceil(num_proj_pix / 2.) * 2, tf.int32)
    
    odd_x = (num_proj_pix-img_size_x)%2
    odd_y = (num_proj_pix-img_size_y)%2
        
    padx = (num_proj_pix-img_size_x)//2
    pady = (num_proj_pix-img_size_y)//2
    
    if integrate_vae:
        paddings = [[0,0],[padx, padx+odd_x], [pady, pady+odd_y],[0,0]]
    else:
        if dim == 3:
            paddings = [[padx, padx+odd_x], [pady, pady+odd_y],[0,0]]
        elif dim == 2:
            paddings = [[padx, padx+odd_x], [pady, pady+odd_y]]
    phantom = tf.pad(phantom, paddings, "CONSTANT")
    return(phantom)

def project_tf_low_mem(phantom,theta, pad = False):
    '''
    Parameters
    ----------
    phantom : TYPE
        img_size_x x img_size_y x img_size_z
    angles : TYPE
        DESCRIPTION.

    Returns
    -------
    sino : TYPE
        DESCRIPTION.

    '''
    if pad:
        phantom = pad_phantom(phantom)
        
    num_angles = len(theta)
    sino_list = []
    for i in range(num_angles):
        img_rot = tfa.image.rotate(phantom,-theta[i], 
                                   interpolation='bilinear',
                                   # interpolation='nearest',
                                   fill_mode='constant',
                                   )
        row = tf.math.reduce_sum(img_rot,0)
        sino_list.append(row)
    sino = tf.stack(sino_list)
    return sino

def project_tf_fast(phantom,theta,pad = False, dim=3,
                    integrate_vae=False):
    '''
    phantom is img_size_x x img_size_y x img_size_z
    or phantom is img_size_x x img_size_y
    
    for integrate_vae option:
    phantom is batch_size x img_size_x x img_size_y x 1
    '''
    
    num_angles = len(theta)

    if pad:
        phantom = pad_phantom(phantom, dim=dim, 
                              integrate_vae=integrate_vae)
    
    '''
    img_size_x = phantom.shape[0]
    img_size_y = phantom.shape[1]
    img_size_z = phantom.shape[2]
    '''
    
    if dim==2 and not(integrate_vae):
        # add a dummy dimension for 3d
        phantom = tf.expand_dims(phantom,-1)
    
    if integrate_vae:
        imgs = tf.transpose(phantom, perm=[3,1,2,0])
        imgs = tf.repeat(imgs, num_angles, axis=0)
    else:
        # imgs is num_angles x img_size_x x img_size_y x img_size_z
        imgs = tf.repeat(tf.expand_dims(phantom,0), num_angles, axis=0)
    
    imgs_rot = tfa.image.rotate(imgs,-theta)
    sino = tf.math.reduce_sum(imgs_rot,1)
    
    if integrate_vae:
        # rotate back
        sino = tf.transpose(sino, perm=[2,0,1])
        
        # add back dummy dimension
        sino = tf.expand_dims(sino, axis=-1)
        
    return(sino)
