#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 14:19:18 2022

@author: amendoz1
"""

# input: list of filepaths to h5
# output to save: numpy array, theta, num_proj_pix

import argparse
import numpy as np
import h5py
import glob
import time
from ctvae.helper_functions import create_folder

def find_data(name):
    if name[-5:-1] == '/dat':
        return name
    
def find_theta(name):
    if name[-6:-1] == '/thet':
        return name

def main():
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('-i','--input_path', type=str, help='path to folder with hdf5 data files', default='real_data')
    parser.add_argument('-o','--output_path', type=str, help='path to folder to place output files in', default='dataset_real')
    args = parser.parse_args()
    
    input_path = args.input_path
    save_path = args.output_path
    create_folder(save_path)

    obj_vec = np.sort(glob.glob(input_path + '/*.h5'))

    x_train_sinograms = []
    for ind,obj in enumerate(obj_vec):
        f = h5py.File(obj, 'r')
        data_path = f.visit(find_data)
        num_z = f[data_path][:,:,:].shape[0]
        # temp = f[data_path][num_z//2-5:num_z//2+5,:,:]
        temp = f[data_path][num_z//2-1:num_z//2+0,:,:]
        if ind==0:
            num_proj_pix = temp.shape[-1]
            theta_path = f.visit(find_theta)
            theta = f[theta_path][:]
        x_train_sinograms.append(temp)
        f.close
        pass
    x_train_sinograms = np.concatenate(x_train_sinograms)


    np.save(save_path + '/x_train_sinograms.npy', x_train_sinograms)

    np.save(save_path + '/dataset_parameters.npy', np.array([theta,
                                                       num_proj_pix], dtype = np.object))


    '''
    # uncomment to test a small dataset
    np.save(save_path + '/x_train_sinograms.npy', x_train_sinograms[:,0:10,1500:1550])
    num_proj_pix = 50
    np.save(save_path + '/dataset_parameters.npy', np.array([theta[0:10],
                                                       num_proj_pix], dtype = np.object))
    '''
    
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Total time was ' + str((end_time-start_time)/60) + ' minutes.')