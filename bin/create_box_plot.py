#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:09:07 2022

@author: vganapa1
"""

import numpy as np
import matplotlib.pyplot as plt

all_trials = np.arange(0,11)

final_ave_merit_random_vec = []
final_ave_merit_uniform_vec = []

for trial in all_trials:
    if trial==0:
        save_path_random = 'foam_pvae_noise_4_ex_3_patt_20_algos'
    else:
        save_path_random ='foam_pvae_noise_4_ex_3_patt_20_algos_' + str(trial)
    save_path_uniform = save_path_random + '_uniform'

    final_ave_merit_random = np.load(save_path_random + '/final_ave_merit.npy') # comparison of ground truth to... (...noisy full sinogram; noisy partial, PVAE) x metric (MSE,SSIM,PSNR)
    final_ave_merit_uniform = np.load(save_path_uniform + '/final_ave_merit.npy') # comparison of ground truth to... (...noisy full sinogram; noisy partial, PVAE) x metric (MSE,SSIM,PSNR)
    
    final_ave_merit_random_vec.append(final_ave_merit_random)
    final_ave_merit_uniform_vec.append(final_ave_merit_uniform)

# num_trials x comparison of ground truth to... (...noisy full sinogram; noisy partial, PVAE) x metric (MSE,SSIM,PSNR)
final_ave_merit_random_vec = np.stack(final_ave_merit_random_vec,axis=0)
final_ave_merit_uniform_vec = np.stack(final_ave_merit_uniform_vec, axis=0)

# Organize data

# rows: trials
# columns: gridrec full, gridrec partial uniform, gridrec random, PVAE uniform, PVAE random
# depth: MSE, SSIM, PSNR



final_vals = np.zeros([len(all_trials),5,3])
final_vals[:,0] = final_ave_merit_random_vec[:,0,:] # gridrec full
final_vals[:,1] = final_ave_merit_uniform_vec[:,1,:] # gridrec partial uniform
final_vals[:,2] = final_ave_merit_random_vec[:,1,:] # gridrec random
final_vals[:,3] = final_ave_merit_uniform_vec[:,2,:] # PVAE uniform
final_vals[:,4] = final_ave_merit_random_vec[:,2,:] # PVAE random

# Find the median trial for MSE, random
print('median trial for MSE, random')
print(all_trials[final_vals[:, 4, 0]==np.median(final_vals[:, 4, 0])])

# Find the median trial for MSE, uniform
print('median trial for MSE, uniform')
print(all_trials[final_vals[:, 3, 0]==np.median(final_vals[:, 3, 0])])


# Find the median trial for PSNR, random
print('median trial for PSNR, random')
print(all_trials[final_vals[:, 4, 2]==np.median(final_vals[:, 4, 2])])

# Find the median trial for PSNR, uniform
print('median trial for PSNR, uniform')
print(all_trials[final_vals[:, 3, 2]==np.median(final_vals[:, 3, 2])])


# Find the median trial for SSIM, random
print('median trial for SSIM, random')
print(all_trials[final_vals[:, 4, 1]==np.median(final_vals[:, 4, 1])])

# Find the median trial for SSIM, uniform
print('median trial for SSIM, uniform')
print(all_trials[final_vals[:, 3, 1]==np.median(final_vals[:, 3, 1])])

# Make box plots
# boxplot is drawn for every column

# (a) gridrec full
# (b) gridrec partial uniform
# (c) gridrec random
# (d) PVAE uniform
# (e) PVAE random

plt.figure()
plt.title('MSE')
plt. boxplot(final_vals[:, :, 0], whis=100,
             labels=['(a)', '(b)', '(c)', '(d)', '(e)'])
plt.savefig('MSE_boxplot.png', 
            bbox_inches="tight", dpi=300)

plt.figure()
plt.title('SSIM')
plt. boxplot(final_vals[:,:,1], whis=100,
             labels=['(a)', '(b)', '(c)', '(d)', '(e)'])
plt.savefig('SSIM_boxplot.png', 
            bbox_inches="tight", dpi=300)

plt.figure()
plt.title('PSNR')
plt. boxplot(final_vals[:,:,2], whis=100,
             labels=['(a)', '(b)', '(c)', '(d)', '(e)'])
plt.savefig('PSNR_boxplot.png', 
            bbox_inches="tight", dpi=300)

# MSE, SSIM, PSNR
# noisy, full sinogram
# [6.94863218e-03 9.33453737e-01 2.17436961e+01]
# noisy, partial sinogram
# [ 0.02464845  0.37429312 16.54751918]
# P-VAE from noisy, partial sinogram
# [8.97230689e-03 6.62524878e-01 2.05639408e+01]