#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 18:33:54 2022
@author: vganapa1
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

np.random.seed(0)

def create_all_masks(x_train_sinograms=None, 
                     num_angles=None,
                     save_path = None,
                     poisson_noise_multiplier = 1e3, # poisson noise multiplier, higher value means higher SNR
                     num_sparse_angles = 10, # number of angles to image per sample (dose remains the same)
                     random = False, # If True, randomly pick angles
                     reg = np.finfo(np.float32).eps.item(),
                     real_data = False,
                     train = False,
                     truncate_dataset = 100,
                     toy_masks = False,
                     **kwargs):

    x_train_sinograms = x_train_sinograms[:truncate_dataset]
    x_train_sinograms[x_train_sinograms<0]=0
    num_examples = len(x_train_sinograms)

    ### Create the masks ###
    if train:
        if toy_masks:

            all_masks = tf.constant([[1,0], # ambiguous mask
                                     [0,1], # unambiguous mask
                                     [1,0],
                                     [0,1]], dtype=tf.float32)
            
            all_masks = np.tile(all_masks, (num_examples//4,1))

        else:
            all_masks = []
            
    
            
            for ind in range(num_examples):
                if random:
                    sparse_angles = tf.random.shuffle(tf.range(num_angles))[:num_sparse_angles]
                else: 
                    # uniformly distribute, but choose a random starting index
                    # start_ind = np.random.randint(0,num_angles)
                    start_ind = 0 # deterministic starting index
                    spacing = np.ceil(num_angles/num_sparse_angles)
                    end_ind = start_ind + spacing*num_sparse_angles
                    all_inds = tf.range(start_ind,end_ind,spacing)
                    sparse_angles = tf.cast(all_inds%num_angles, tf.int32)
                mask = tf.expand_dims(tf.reduce_sum(tf.one_hot(sparse_angles,num_angles,axis=0),axis=1),axis=0)
                mask = mask/num_sparse_angles
                all_masks.append(mask)
            all_masks = tf.concat(all_masks,axis=0)
            

        
        # print(all_masks.shape)
        # print(mask.shape)
        # print(all_masks)
        
        # shape of all_masks is num_examples x num_angles
        np.save(save_path + '/all_masks.npy', all_masks)
    else:
        all_masks = np.load(save_path + '/all_masks.npy')
    
    ### Make the corresponding sparse sinograms ###
    if train:
        all_proj_samples = []
        for ind in range(num_examples):
            proj = x_train_sinograms[ind]
            proj_masked = proj*tf.expand_dims(all_masks[ind], axis=1)
        
            if real_data:
                proj_sample = proj_masked
            else:
                '''
                # add Poisson-like noise
                proj_dist = tfd.Normal(loc = proj_masked, \
                                        scale = reg + tf.sqrt(proj_masked/poisson_noise_multiplier))
                proj_sample = proj_dist.sample() 
                '''
                # add Poisson noise
    
                proj_dist = tfd.Poisson(proj_masked*poisson_noise_multiplier)
                proj_sample = proj_dist.sample()/poisson_noise_multiplier  
                
                #
                
            all_proj_samples.append(proj_sample)
            
        # shape is num_examples x num_angles x num_proj_pix
        all_proj_samples = np.stack(all_proj_samples)
        np.save(save_path + '/all_proj_samples.npy', all_proj_samples)
    else:
        all_proj_samples = np.load(save_path + '/all_proj_samples.npy')

    return(all_masks, all_proj_samples)
