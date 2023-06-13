#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:53:33 2022

@author: vganapa1
"""
import numpy as np
import xdesign as xd 
from .helper_functions import create_sinogram, get_images, create_folder, get_sinograms
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def create_dataset(N_PIXEL = 128,
                   SIZE_LOWER = 0.01,
                   SIZE_UPPER = 0.2,
                   GAP = 0,
                   num_train = 100,
                   save_name = 'foam_training'):

    np.random.seed(0)
    x_train = []

    for i in range(num_train):
        phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, porosity=np.random.rand())
        discrete = xd.discrete_phantom(phantom, N_PIXEL)
        x_train.append(discrete)
        print(i)
    x_train = np.stack(x_train, axis=0)
    np.save(save_name + '.npy', x_train)
    print('Dataset saved as ' + save_name + '.npy')
    return(x_train)

def preformat_data(theta = np.linspace(0, np.pi, 20, endpoint=False), # projection angles
                   save_path = 'dataset_foam_test',
                   truncate_dataset = 100,
                   img_type = 'foam', # 'mnist' or 'foam'
                   ):
    
    create_folder(save_path)
    
    # pull images that are normalized from 0 to 1
    # 0th dimension should be the batch dimension
    # 1st and 2nd dimensions should be spatial x and y coords, respectively
    x_train_imgs = get_images(img_type = img_type)
    
    x_train_imgs = x_train_imgs[0:truncate_dataset]
    x_train_sinograms = []
    
    for b in range(x_train_imgs.shape[0]):
        print(b)
        img = x_train_imgs[b]
        sinogram = create_sinogram(img, theta)
        x_train_sinograms.append(np.expand_dims(sinogram, axis=0))

    x_train_sinograms = np.concatenate(x_train_sinograms, axis=0)
    
    num_proj_pix = x_train_sinograms.shape[-1]
    
    x_train_sinograms[x_train_sinograms<0]=0
    
    np.save(save_path + '/x_train_sinograms.npy', x_train_sinograms)
    np.save(save_path + '/dataset_parameters.npy', np.array([theta,
                                                   num_proj_pix], dtype = np.object))

    np.save(save_path + '/x_size.npy', x_train_imgs.shape[1])
    np.save(save_path + '/y_size.npy', x_train_imgs.shape[2])
    
    print("Shape of sinograms: ", x_train_sinograms.shape)
    print("Shape of original training images: ", x_train_imgs.shape)
    return(x_train_sinograms, num_proj_pix)

def create_masks(input_path = 'dataset_foam_test', # dataset to process
                 poisson_noise_multiplier = (2**16-1)*0.41, # poisson noise multiplier, higher value means higher SNR
                 num_sparse_angles = 5, # number of angles to image per sample (dose remains the same)
                 save_tag = 'pnm2e4_angles5', # output is saved in input_path/save_tag
                 save_tag_masks = None, # if None, generate masks, otherwise use the masks saved in input_path/save_tag_masks
                 random = False,
                 ):
        
    ### SET/LOAD VALUES ###
    
    reg = np.finfo(np.float32).eps.item()
    x_train_sinograms, theta, num_proj_pix = get_sinograms(input_path)
    num_examples = len(x_train_sinograms)
    num_angles = len(theta)
    
    
    ### Create the masks ###
    
    if save_tag_masks is None:
    
        all_masks = []
        for ind in range(num_examples):
            if random:
                sparse_angles = tf.random.shuffle(tf.range(num_angles))[:num_sparse_angles]
            else: # uniformly distribute, but choose a random starting index
                start_ind = np.random.randint(0,num_angles)
                spacing = np.ceil(num_angles/num_sparse_angles)
                end_ind = start_ind + spacing*num_sparse_angles
                all_inds = tf.range(start_ind,end_ind,spacing)
                sparse_angles = tf.cast(all_inds%20, tf.int32)
            mask = tf.expand_dims(tf.reduce_sum(tf.one_hot(sparse_angles,num_angles,axis=0),axis=1),axis=0)
            mask = mask/num_sparse_angles
            all_masks.append(mask)
        all_masks = tf.concat(all_masks,axis=0)
        create_folder(input_path + '/' + save_tag)
    else:
        all_masks = np.load(input_path + '/' + save_tag_masks + '/all_masks.npy')
        
    np.save(input_path + '/' + save_tag + '/all_masks.npy', all_masks)
    
    ### Make the corresponding sparse sinograms ###
    all_proj_samples = []
    for ind in range(num_examples):
        proj = x_train_sinograms[ind]
        proj_masked = proj*tf.expand_dims(all_masks[ind], axis=1)
    
        # add Poisson-like noise
        proj_dist = tfd.Normal(loc = proj_masked, \
                                scale = reg + tf.sqrt(proj_masked/poisson_noise_multiplier))
            
        proj_sample = proj_dist.sample()  
        all_proj_samples.append(proj_sample)
    all_proj_samples = np.stack(all_proj_samples)
    np.save(input_path + '/' + save_tag + '/all_proj_samples.npy', all_proj_samples)
    return(all_masks, all_proj_samples)