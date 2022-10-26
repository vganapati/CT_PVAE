#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:05:38 2022

@author: vganapa1
"""
import sys
import time
import tomopy

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from ctvae.toy_mcmc_v2_functions import load_batch_mcmc, joint_log_prob
from ctvae.helper_functions import toy_dist
tfd = tfp.distributions
tfb = tfp.bijectors

### COMMAND LINE INPUTS ###

parser = argparse.ArgumentParser(description='Get command line args')

parser.add_argument('--save_path', action='store',
                    help='path to save output, same path as the VAE')

parser.add_argument('-s', type=int, action='store', dest='number_of_steps', 
                    help='number of training iterations', default = 200000)
    
parser.add_argument('-b', type=int, action='store', dest='burnin', 
                    help='number of training iterations', default = 50000)

args = parser.parse_args()

### END COMMAND LINE INPUTS ###

### INPUTS ###
example_num = 0
ground_truth = 'toy_training.npy'
input_path = 'dataset_toy'
N_PIXEL = 2 # x_size
poisson_noise_multiplier = 1e3
theta = np.array([0,np.pi/2], dtype=np.float32)
save_path = args.save_path
leapfrog_steps = 5
step_size = 6.5e-2
number_of_steps = args.number_of_steps
burnin = args.burnin
### END INPUTS ###


all_masks = np.load(save_path + '/all_masks.npy')
all_proj_samples = np.load(save_path + '/all_proj_samples.npy')


mask,proj_sample = load_batch_mcmc(all_masks,
                              all_proj_samples,
                              example_num)


ground_truth_reshape = tf.reshape(ground_truth,[-1])


# probability distribution P(O)
# mix_prob = 0.3 # probability of the 0th distribution
# conc_0 = np.array([0.35580334, 0.94963009, 0.60227688, 0.43061459], dtype=np.float32)
# conc_1 = np.array([0.00390356, 0.44335424, 0.83152378, 0.52733124], dtype=np.float32)
bimix_dist = toy_dist()
 




#############
# Tomopy reconstruct partial noisy sinogram

used_angles = mask>0
used_mask = np.expand_dims(mask[used_angles], axis=1)


# # Using tomopy
# initial_guess = tomopy.recon(np.expand_dims(proj_sample[used_angles]/used_mask,axis=0),\
#                       theta[used_angles], center=None, algorithm='fbp', sinogram_order=True)
# initial_guess = tf.squeeze(initial_guess)
# initial_guess = tf.reshape(initial_guess,[-1])
# initial_guess = initial_guess/tf.reduce_sum(initial_guess)


initial_guess = tf.ones([N_PIXEL,N_PIXEL],dtype=tf.float32)
initial_guess = initial_guess/tf.reduce_sum(initial_guess)
initial_guess = tf.reshape(initial_guess,[-1])


# Set the chain's start state.
initial_chain_state = [initial_guess]


# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.IteratedSigmoidCentered()       # Maps R to simplex
]


# Define a closure over our joint_log_prob.

unnormalized_posterior_log_prob = lambda O: joint_log_prob(O, 
                                                           proj_sample[used_angles], # measurements
                                                           bimix_dist,
                                                           theta[used_angles],
                                                           mask[used_angles],
                                                           N_PIXEL,
                                                           poisson_noise_multiplier,
                                                           )
    
    




# Defining the HMC



hmc = tfp.mcmc.SimpleStepSizeAdaptation(tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=leapfrog_steps,
        step_size=step_size,
        state_gradients_are_stopped=False,
        store_parameters_in_results=False),
    bijector=unconstraining_bijectors), num_adaptation_steps=400)

start_time = time.time()

# Sampling from the chain.
[
    posterior_prob
], kernel_results = tfp.mcmc.sample_chain(
    num_results=number_of_steps,
    num_burnin_steps=burnin,
    current_state=initial_chain_state,
    kernel=hmc,
    trace_fn=(lambda current_state, kernel_results: kernel_results))

burned_prob_trace = posterior_prob[burnin:]

end_time = time.time()

np.save(save_path + '/posterior_prob_trace.npy', posterior_prob)

print('Total time elapsed is (minutes): ' + str((end_time-start_time)/60))
##############

# print('burned prob trace')
# print(burned_prob_trace)

### Output Figures ###
for pixel_ind in range(N_PIXEL**2):
    plt.figure(figsize=[12.5, 4])
    plt.title('MCMC result, True value: ' + str(ground_truth_reshape[pixel_ind]))
    plt.hist(burned_prob_trace[:,pixel_ind], bins=25, histtype="stepfilled", density=True)
    plt.plot(ground_truth_reshape[pixel_ind]*np.ones(50),np.arange(50))
    plt.savefig(save_path + '/pixel_mcmc_' + str(pixel_ind) + '.png')

