#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:09:07 2022

@author: vganapa1
"""

import tomopy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa

from ctvae.forward_functions import project_tf_fast

tfd = tfp.distributions
tfb = tfp.bijectors



def load_batch_mcmc(all_masks,
                    all_proj_samples,
                    example_num):
    mask = all_masks[example_num]
    proj_sample = all_proj_samples[example_num]

    return(mask,proj_sample)

def get_likelihood(O,
                   theta,
                   mask,
                   N_PIXEL,
                   poisson_noise_multiplier,
                   sqrt_reg=np.finfo(np.float32).eps.item(),
                   reshape=True,
                   ):
    
    if reshape:
        O = tf.reshape(O, [N_PIXEL,N_PIXEL])
    proj = project_tf_fast(O,theta,pad = False, dim=2,integrate_vae=False)
    proj = tf.squeeze(proj,axis=-1)
    proj_masked = proj*tf.expand_dims(mask, axis=-1)


    # if theta[0] == 0:
    #     proj = tf.reduce_sum(O, axis=0)
    # else:
    #     proj = tf.reduce_sum(O, axis=1)[::-1]
    # proj = tf.expand_dims(proj, axis=0)

    # tf.print('O')
    # tf.print(O)
    # tf.print('proj')
    # tf.print(proj)
    
    # proj = tomopy.project(obj, theta, center=None, emission=True, pad=False, sinogram_order=True)
    # likelihood = tfd.Normal(proj_masked, (tf.sqrt(proj_masked))/poisson_noise_multiplier+sqrt_reg)

    likelihood = tfd.Poisson(proj_masked*poisson_noise_multiplier, force_probs_to_zero_outside_support=False)

    # likelihood prob
    # tf.reduce_sum(likelihood.log_prob(M*poisson_noise_multiplier)
    return(likelihood)

@tf.function
def joint_log_prob(O, 
                   M,
                   bimix_dist,
                   theta,
                   mask,
                   N_PIXEL,
                   poisson_noise_multiplier,
                   ):
    """
    Joint log probability optimization function.
        
    Args:
      occurrences: An array of binary values (0 & 1), representing 
                   the observed frequency
      prob_A: scalar estimate of the probability of a 1 appearing 
    Returns: 
      sum of the joint log probabilities from all of the prior and conditional distributions
    """  
    O = tf.maximum(O, np.finfo(np.float32).tiny)
    likelihood  = get_likelihood(O,theta, mask, N_PIXEL, poisson_noise_multiplier)
    # tf.print('O')
    # tf.print(O)
    # tf.print('likelihood')
    # tf.print(tf.reduce_sum(likelihood.log_prob(M*poisson_noise_multiplier)))

    return (
        bimix_dist.log_prob(O) + tf.reduce_sum(likelihood.log_prob(M*poisson_noise_multiplier
                                                                   ))
    )
    # return (
    #     bimix_dist.log_prob(O) + tf.reduce_sum(likelihood.log_prob(M))
    # )

