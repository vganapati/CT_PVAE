#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 16:37:37 2022

@author: vganapa1
"""

from ctvae.helper_functions import get_sinograms, plot_single_example
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--en', type=int, action='store', dest='example_num', 
                    help='example index for visualization', default = 0)

args = parser.parse_args()


input_path = 'dataset_foam'
num_patt = 20
save_path_random = 'foam_pvae_noise_4_ex_3_patt_' + str(num_patt) + '_algos'
save_path_uniform = 'foam_pvae_noise_4_ex_3_patt_' + str(num_patt) + '_algos_uniform'
example_num = args.example_num


# full sinogram 
x_train_sinograms, theta, num_proj_pix = get_sinograms(input_path)
x_train_sinogram = x_train_sinograms[example_num]
plot_single_example(x_train_sinogram, 'full_sinogram', input_path, example_num, vmin=None, vmax=None)
vmax = np.max(x_train_sinogram)
vmin = np.min(x_train_sinogram)

# partial sinogram random
all_proj_samples = np.load(save_path_random + '/all_proj_samples.npy')
proj_random = num_patt*all_proj_samples[example_num] # multiplying by num_patt removes normalization
plot_single_example(proj_random, 'proj_random', input_path, example_num, vmin=vmin, vmax=vmax)

# partial sinogram uniform
all_proj_samples = np.load(save_path_uniform + '/all_proj_samples.npy')
proj_uniform = num_patt*all_proj_samples[example_num]
plot_single_example(proj_uniform, 'proj_uniform', input_path, example_num, vmin=vmin, vmax=vmax)

# ground truth
prefix = input_path[8:]
ground_truth = np.load(prefix + '_training.npy')[example_num]
padding = (num_proj_pix-ground_truth.shape[0])//2
ground_truth = np.pad(ground_truth,((padding,padding),(padding,padding)))
plot_single_example(ground_truth, 'ground_truth', input_path, example_num, vmin=None, vmax=None)