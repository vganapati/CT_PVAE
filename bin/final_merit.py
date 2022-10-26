#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:23:22 2022

@author: vganapa1
"""
import argparse
import tomopy
import numpy as np
import sys
import tensorflow_probability as tfp
tfd = tfp.distributions
from ctvae.helper_functions import get_sinograms, crop, compare

### INPUTS ###

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--input_path', action='store',
                    help='path to folder containing training data')

parser.add_argument('--save_path', action='store',
                    help='path to save output')

parser.add_argument('--pnm', type=float, action='store', dest='poisson_noise_multiplier',
                    help='poisson noise multiplier, higher value means higher SNR')

args = parser.parse_args()

input_path = args.input_path
save_path = args.save_path
poisson_noise_multiplier = args.poisson_noise_multiplier

### END OF INPUTS ####

prefix = input_path[8:]
algorithm = 'gridrec' # 'sirt', 'tv', 'gridrec', 'fbp'

# load in final reconstructions from the P-VAE

reconstruction_final = np.squeeze(np.load(save_path + '/reconstruction_final.npy'), axis=-1)
x_size = reconstruction_final.shape[1]
y_size = reconstruction_final.shape[2]
truncate_dataset = reconstruction_final.shape[0]
# load in the ground truth and truncate to the length of reconstruction_final

ground_truth = np.load(prefix + '_training.npy')[0:truncate_dataset]

# reconstruction from noisy version of the full sinogram

x_train_sinograms, theta, num_proj_pix = get_sinograms(input_path)
x_train_sinograms = x_train_sinograms[0:truncate_dataset] 

actual_sinogram_dist = tfd.Poisson(x_train_sinograms*poisson_noise_multiplier)
actual_sinogram_noisy = actual_sinogram_dist.sample()/poisson_noise_multiplier  

recon0 = tomopy.recon(actual_sinogram_noisy, theta, center=None, algorithm=algorithm, sinogram_order=True)

recon0 = crop(recon0,x_size,y_size,ignore_dim_0=True)

recon0 = np.minimum(recon0,1)
recon0 = np.maximum(recon0,0)


# reconstruction from noisy, partial sinogram

partial_noisy_sinograms = np.load(save_path + '/all_proj_samples.npy')
all_masks = np.load(save_path + '/all_masks.npy')

   

used_angles = all_masks>0
recon1_all = []



for ind in range(truncate_dataset):
    partial_noisy_sinogram = partial_noisy_sinograms[ind]
    used_mask = np.expand_dims(all_masks[ind][used_angles[ind]], axis=1)
    recon1 = tomopy.recon(np.expand_dims(partial_noisy_sinogram[used_angles[ind]]/used_mask, axis=1),\
                          theta[used_angles[ind]], center=None, algorithm=algorithm, sinogram_order=False)
    recon1 = np.squeeze(recon1,axis=0)
    recon1 = crop(recon1,x_size,y_size)
    recon1_all.append(recon1)

recon1_all = np.stack(recon1_all,axis=0)
recon1_all = np.minimum(recon1_all,1)
recon1_all = np.maximum(recon1_all,0)

# evaluate the merit against the ground truth for all examples in the dataset

merit_0_all = []
merit_1_all = []
merit_2_all = []

for ind in range(truncate_dataset):
    merit_0 = compare(ground_truth[ind], recon0[ind]) # merit is MSE, SSIM, PSNR
    merit_1 = compare(ground_truth[ind], recon1_all[ind])
    merit_2 = compare(ground_truth[ind], reconstruction_final[ind])
    
    merit_0_all.append(merit_0)
    merit_1_all.append(merit_1)
    merit_2_all.append(merit_2)
    
merit_0_all = np.stack(merit_0_all, axis=0)
merit_1_all = np.stack(merit_1_all, axis=0)
merit_2_all = np.stack(merit_2_all, axis=0)

print('MSE, SSIM, PSNR')
print('noisy, full sinogram')
print(np.mean(merit_0_all,axis=0))
print('noisy, partial sinogram')
print(np.mean(merit_1_all,axis=0))
print('P-VAE from noisy, partial sinogram')
print(np.mean(merit_2_all,axis=0))

final_ave_merit = np.array([np.mean(merit_0_all,axis=0),np.mean(merit_1_all,axis=0),np.mean(merit_2_all,axis=0)])
np.save(save_path + '/final_ave_merit.npy', final_ave_merit)





