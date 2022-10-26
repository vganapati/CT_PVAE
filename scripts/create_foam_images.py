#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 10:23:09 2022

@author: vganapa1

Creates synthetic dataset of foam objects

"""

import argparse
import numpy as np
import xdesign as xd 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-n','--num-train', type=int, help='number of points', default=15000)
    args = parser.parse_args()
    
    ### INPUTS ###

    N_PIXEL = 128 # size of each phantom is N_PIXEL x N_PIXEL

    # parameters to generate the foam phantoms
    SIZE_LOWER = 0.01
    SIZE_UPPER = 0.2
    GAP = 0

    num_train = args.num_train # number of phantoms created

    ### END OF INPUTS ###

    np.random.seed(0)
    x_train = []

    for i in range(num_train):
        phantom = xd.Foam(size_range=[SIZE_UPPER, SIZE_LOWER], gap=GAP, porosity=np.random.rand())
        discrete = xd.discrete_phantom(phantom, N_PIXEL)
        x_train.append(discrete)
        print(i)
    x_train = np.stack(x_train, axis=0) # shape is num_train x N_PIXEL x N_PIXEL
    np.save('foam_training.npy', x_train)
    del x_train