#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:59:01 2022

@author: vganapa1
"""

from ctvae.helper_functions import toy_dist

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-n','--num-train', type=int, help='number of points', default=15000)
    args = parser.parse_args()
    
    ### INPUTS ###

    discrete = True # overrides all other inputs
    N_PIXEL = 2 # size of each phantom is N_PIXEL x N_PIXEL
    # mix_prob = 0.3 # probability of the 0th distribution
    num_train = args.num_train # number of phantoms created

    ### END OF INPUTS ###
    
    if discrete:
        x_train_0 = np.array([[1,2],[3,4]])/10
        x_train_1 = np.array([[3,4],[1,2]])/10
        x_train = np.stack((x_train_0,x_train_1),axis=0)
        x_train = np.repeat(x_train, repeats=2, axis=0)
        x_train = np.tile(x_train, (10000,1,1))
        np.save('toy_discrete2_training.npy', x_train)
    else:
        bimix_dist = toy_dist()
        x_train = bimix_dist.sample(num_train)
        
        
        x_train = tf.reshape(x_train, [num_train,N_PIXEL,N_PIXEL]) # shape is num_train x N_PIXEL x N_PIXEL

        np.save('toy_training.npy', x_train)
    
    
    