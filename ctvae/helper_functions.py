#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:59:53 2021

@author: vganapa1
"""
import tomopy
from .forward_functions import project_tf_fast
import tensorflow as tf
import numpy as np
import os
import tensorflow_probability as tfp
import tensorflow_addons as tfa

tfd = tfp.distributions


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

import matplotlib.pyplot as plt

def create_folder(save_path=None,**kwargs):
    try: 
        os.makedirs(save_path)
    except OSError:
        if not os.path.isdir(save_path):
            raise

def create_sinogram(img, theta, pad=True):
    # multiprocessing.freeze_support()
    phantom = np.expand_dims(img,axis=0)
    proj = tomopy.project(phantom, theta, center=None, emission=True, pad=pad, sinogram_order=False)
    proj = np.squeeze(proj,axis=1)
    return proj

def get_images(img_type = 'mnist'):
    if img_type == 'mnist':
        mnist = tf.keras.datasets.mnist
        (x_train, _), (_, _) = mnist.load_data()
        
        x_train = x_train / 255.
    else:
        x_train = np.load(img_type + '_training.npy')
    return(x_train)

def get_sinograms(save_path):
    theta,\
    num_proj_pix = np.load(save_path + '/dataset_parameters.npy',
                               allow_pickle = True)
    
    x_train_sinograms = np.load(save_path + '/x_train_sinograms.npy')
    return(x_train_sinograms, theta, num_proj_pix)

def create_tf_dataset(x_train_sinograms,
                      all_proj_samples,
                      all_masks,
                      all_input_encode,
                      truncate_dataset,
                      batch_size,
                      buffer_size,                      
                      num_angles,
                      theta,
                      angles_per_iter,
                      roll = False,
                      **kwargs,
                      ):
    
    # reduce size of dataset to truncate dataset
    x_train_sinograms = x_train_sinograms[0:truncate_dataset]
    train_size = len(x_train_sinograms)
    
    train_ds_0 = tf.data.Dataset.from_tensor_slices(x_train_sinograms)
    train_ds_1 = tf.data.Dataset.from_tensor_slices(all_proj_samples)
    train_ds_2 = tf.data.Dataset.from_tensor_slices(all_masks)
    train_ds_3 = tf.data.Dataset.from_tensor_slices(all_input_encode)
    
    
    
    train_ds = tf.data.Dataset.zip((train_ds_0, train_ds_1, train_ds_2, train_ds_3))
    
    if roll:
        # XXX need to check that roll is done correctly
        random_roll_lambda = lambda x_train_sinogram, proj_sample, mask, input_encode: random_roll(x_train_sinogram, 
                                                                                                   proj_sample, 
                                                                                                   mask,input_encode, 
                                                                                                   num_angles, theta)
        
        train_ds = train_ds.map(random_roll_lambda)
    
    autotune = tf.data.experimental.AUTOTUNE
    
    train_ds_unshuffled = configure_for_performance(train_ds, batch_size, autotune, 
                                         shuffle=False, buffer_size = buffer_size, repeat=False)
    train_ds_unshuffled = iter(train_ds_unshuffled)
    
    train_ds = configure_for_performance(train_ds, batch_size, autotune, 
                                         shuffle=True, buffer_size = buffer_size, repeat=True)
    train_ds = iter(train_ds)

    train_ds_angles = tf.data.Dataset.from_tensor_slices(np.arange(num_angles))
    train_ds_angles = configure_for_performance(train_ds_angles, angles_per_iter, autotune, 
                                         shuffle=True, buffer_size = num_angles, repeat=True)
    train_ds_angles = iter(train_ds_angles)
    
    return(train_ds, train_size, train_ds_unshuffled, train_ds_angles)

def random_roll(x_train_sinogram, proj_sample, mask, input_encode, 
                num_angles, theta,
                num_proj_pix):
    # XXX Check that function is implemented correctly
    roll_length = tf.random.uniform(
                        [1,],
                        minval=0,
                        maxval=num_angles,
                        dtype=tf.int32,
                    )
    rot_angle = tf.gather(theta,roll_length)
    x_train_sinogram = single_roll(x_train_sinogram,
                                   roll_length,
                                   num_angles,
                                   num_proj_pix)
    
    proj_sample = single_roll(proj_sample,
                              roll_length,
                              num_angles,
                              num_proj_pix)
    
    mask = single_roll(mask,
                       roll_length,
                       num_angles,
                       num_proj_pix)
    
    ### XXX STOPPED HERE, check rotation in the correct direction
    input_encode = tfa.image.rotate(input_encode,rot_angle, 
                               interpolation='bilinear',
                               # interpolation='nearest',
                               fill_mode='constant',
                               )
        
    return(x_train_sinogram, proj_sample, mask, input_encode)

def single_roll(x_train_sinogram,
                roll_length,
                num_angles,
                num_proj_pix):
   x_train_sinogram = tf.concat((x_train_sinogram,tf.experimental.numpy.flip(x_train_sinogram,axis=0)),axis=0)
   x_train_sinogram = tf.roll(x_train_sinogram, shift=tf.squeeze(roll_length), axis=0)
   x_train_sinogram = tf.slice(x_train_sinogram,(0,0),(num_angles,num_proj_pix))
   return(x_train_sinogram)

def configure_for_performance(ds, batch_size, 
                              autotune, 
                              shuffle = True,
                              buffer_size = 100,
                              repeat = True):
    if repeat:
        ds = ds.repeat()
        
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)

    ds = ds.batch(batch_size, drop_remainder=True)

    ds = ds.prefetch(buffer_size=autotune)
    return ds

def create_coords(num_angles, # x-dimension
                  num_proj_pix, # y-dimension
                  batch_size,
                  periodic_x = True,
                  periodic_y = False):
    
    # Coordinates, angles x spatial coords
    coords_x_np = (np.arange(0,num_angles) - num_angles/2)/num_angles
    if periodic_x:
        coords_x_np = np.sin(np.pi*(coords_x_np)) # 2*np.pi

    coords_y_np = (np.arange(0,num_proj_pix) - num_proj_pix/2)/num_proj_pix
    if periodic_y:
        coords_y_np = np.sin(np.pi*(coords_y_np)) # 2*np.pi
    
    coords_xm_np, coords_ym_np = np.meshgrid(coords_x_np, coords_y_np, indexing='ij')
    
    coords_xm_np = np.expand_dims(coords_xm_np, axis=-1)
    coords_ym_np = np.expand_dims(coords_ym_np, axis=-1)
    coords_np = np.concatenate((coords_xm_np,coords_ym_np),axis=-1)
    coords_np = np.expand_dims(coords_np,axis=0)
    coords_np = np.repeat(coords_np, batch_size, axis=0)
    
    coords = tf.constant(coords_np, dtype = tf.float32)
    return(coords)

def positive_range(x, offset = np.finfo(np.float32).eps.item()):
    x -= 1
    mask = x<0
    return((tf.exp(tf.clip_by_value(x, -1e10, 10))+offset)*tf.cast(mask, tf.float32) + (x+1)*(1-tf.cast(mask, tf.float32)))

def find_loss_vae_unsup(proj_sample,
                        mask,
                        input_encode,
                        num_proj_pix,
                        x_size,
                        y_size,
                        deterministic,
                        num_blocks,  
                        model_encode,
                        model_decode,
                        poisson_noise_multiplier,
                        sqrt_reg,
                        batch_size,
                        prior,
                        use_normal,
                        training,
                        kl_anneal,
                        kl_multiplier,
                        num_samples = 2,
                        theta = None,
                        angles_i = None,
                        pad=True,
                        ):
    
    '''
    
    Overall process is:

    M ------> z -------> R ------> M   
    
    '''


    ### M --> Z ###
    ### takes input_encode and process through model_encode ###
    skips_val = model_encode((input_encode/300), training = training)       

    ### create conditional latent variable distribution q(z|M) ###
    # all the skip connections are be counted as latent variables as well

    if deterministic:
        q = None
    else:
        q = []
        for i in range(num_blocks+1):
            loc, log_scale = tf.split(skips_val[i], [skips_val[i].shape[-1]//2, skips_val[i].shape[-1]//2], axis=-1, num=None, name='split')
            scale = positive_range(log_scale)
            if use_normal:
                q.append(tfd.Normal(loc=loc, scale=scale+sqrt_reg))
            else:
                q.append(tfd.Beta(positive_range(loc), scale))

    ### z --> R ###
    # sample all the latent variables q(z|X) and process through model_decode
    
    output_dist_vec = []
    log_prob_M_vec = []
    log_prob_M_given_R_vec = []
    log_prob_R_given_z_vec = []
    for s in range(num_samples):
        if deterministic:
            q_sample = skips_val
        else:
            q_sample  = [q[i].sample() for i in range(len(q))]
        
        im_stack_alpha, im_stack_beta = model_decode((q_sample), training = training)
        
        # output a beta distribution, dims: batch_size x image_x x image_y x num_leds
        if use_normal:
            output_dist = tfd.TruncatedNormal(positive_range(im_stack_alpha), positive_range(im_stack_beta),low=0, high=1e10)
            # output_dist = tfd.Normal(positive_range(im_stack_alpha), positive_range(im_stack_beta))
        else:
            output_dist = tfd.Beta(positive_range(im_stack_alpha), positive_range(im_stack_beta))
        output_dist_vec.append(output_dist)
        

        output_sample = output_dist.sample()
        
        if use_normal:
            log_prob_R_given_z = output_dist.log_prob(output_sample)
        else:
            log_prob_R_given_z = output_dist.log_prob(tf.clip_by_value(output_sample, sqrt_reg, 1-sqrt_reg))
        
        '''
        Need to calculate:
        log (P(Measurement | Reconstruction))    
        
        Measurement is proj_sample
        Reconstruction is output_dist
        '''
        
        log_prob_M_given_R = calculate_log_prob_M_given_R(output_sample, # reconstruction
                                                          mask, # measurement parameters
                                                          proj_sample, # measurement
                                                          poisson_noise_multiplier,
                                                          sqrt_reg,
                                                          theta=theta,
                                                          angles_i=angles_i,
                                                          pad=pad,
                                                          )
        
        log_prob_M = tf.reduce_sum(tf.squeeze(log_prob_M_given_R, axis=-1), axis=[0,1,2])\
                     + tf.reduce_sum(tf.squeeze(log_prob_R_given_z, axis=-1), axis=[0,1,2])
 

        
        log_prob_M_given_R_vec.append(tf.reduce_sum(tf.squeeze(log_prob_M_given_R, axis=-1), axis=[0,1,2]))
        log_prob_R_given_z_vec.append(tf.reduce_sum(tf.squeeze(log_prob_R_given_z, axis=-1), axis=[0,1,2]))
        log_prob_M_vec.append(log_prob_M)
    
    # tf.print('log_prob_M_given_R_vec')
    # tf.print(tf.reduce_mean(log_prob_M_given_R_vec))
    
    # tf.print('log_prob_R_given_z_vec')
    # tf.print(tf.reduce_mean(log_prob_R_given_z_vec))        
    
    # VAE objective

    if deterministic:
        kl_divergence = 0
    else:
        kl_divergence = [tf.reduce_sum(tfp.distributions.kl_divergence(q[i], prior[i]),axis=[1,2,3])\
                         for i in range(1,num_blocks+1)] # first skip val (the input), is unused
        kl_divergence = tf.reduce_sum(kl_divergence,axis=0)

    loglik = tf.reduce_mean(log_prob_M_vec,axis=0)
    loss_M_VAE = kl_anneal*kl_multiplier*kl_divergence - loglik    

    return(loss_M_VAE, output_dist_vec, q, q_sample, kl_divergence, loglik, input_encode)

def calculate_log_prob_M_given_R(output_sample, # reconstruction
                                 mask, # measurement parameters
                                 proj_sample, # measurement
                                 poisson_noise_multiplier,
                                 sqrt_reg,
                                 theta=None,
                                 angles_i=None,
                                 pad=True,
                                 ):

    # output_sample is batch_size x x_size x y_size x 1
    # print(output_sample.shape) # (10, 128, 128, 1)

    # angles_i = tf.cast(angles_i,tf.int32)
    if angles_i is not None:
        # theta = theta[angles_i]
        # mask = mask[:,angles_i]
        # proj_sample = proj_sample[:,angles_i]
        
        theta = tf.cast(tf.gather(theta, angles_i, axis=0), tf.float32)
        mask = tf.gather(mask, angles_i,axis=1)
        proj_sample = tf.gather(proj_sample, angles_i, axis=1)
        
    proj = project_tf_fast(output_sample,theta,pad = pad, dim = 2, integrate_vae=True)    
    proj_masked = proj*tf.expand_dims(tf.expand_dims(mask, axis=-1),axis=-1)


    # add Poisson-like noise
    proj_dist = tfd.Normal(loc = proj_masked, \
                            scale = sqrt_reg + tf.sqrt(proj_masked/poisson_noise_multiplier + sqrt_reg))
    proj_sample_expand = tf.expand_dims(proj_sample,axis=-1)
    
    return(proj_dist.log_prob(proj_sample_expand))

def plot(save_path, 
         vec,
         title,
         save_name,
         ):

    plt.figure()
    plt.title(title)
    plt.plot(vec)
    plt.savefig(save_path + '/' + save_name + '.png')

def plot_single_example(img, save_name, 
                        input_path,
                        example_num,
                        vmin=None, vmax=None):
    plt.figure(figsize=[10,10])
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.savefig(input_path + '/' + save_name + '_ex_' + str(example_num) + '.png', 
                bbox_inches="tight", dpi=300, pad_inches=0.0)

def compare(recon0, recon1, verbose=False):

    mse_recon = mean_squared_error(recon0, recon1)
    # np.mean((recon0-recon1)**2)
    
    small_side = np.min(recon0.shape)
    if small_side<7:
        if small_side%2: # if odd
            win_size=small_side
        else:
            win_size=small_side-1
    else:
        win_size=None

    ssim_recon = ssim(recon0, recon1,
                      data_range=recon0.max() - recon0.min(), win_size=win_size)
    
    
    psnr_recon = peak_signal_noise_ratio(recon0, recon1,
                      data_range=recon0.max() - recon0.min())
    
    if verbose:
        err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    return(mse_recon, ssim_recon, psnr_recon)

def crop(img_2d, final_x, final_y, ignore_dim_0=False):
    if ignore_dim_0:
        _,x,y = img_2d.shape
    else:
        x,y = img_2d.shape
    remain_x = final_x % 2
    remain_y = final_y % 2
    if ignore_dim_0:
        return(img_2d[:,x//2 - final_x//2:x//2+final_x//2+remain_x, y//2-final_y//2:y//2+final_y//2+remain_y])
    else:
        return(img_2d[x//2 - final_x//2:x//2+final_x//2+remain_x, y//2-final_y//2:y//2+final_y//2+remain_y])

def evaluate_sinogram(actual_sinogram, 
                      computed_sinogram, 
                      partial_noisy_sinogram, 
                      mask,
                      theta,
                      final_x, # actual size of reconstruction (tomopy reconstructs blank space)
                      final_y, # actual size of reconstruction (tomopy reconstructs blank space)
                      algorithm = 'sirt', # 'sirt', 'tv', 'gridrec', 'fbp'
                      ):
    
    # Tomopy reconstruct actual
    
    recon0 = tomopy.recon(np.expand_dims(actual_sinogram, axis=1), theta, center=None, algorithm=algorithm, sinogram_order=False)
    recon0 = np.squeeze(recon0)
    
    
    # Tomopy reconstruct prediction
    
    recon1 = tomopy.recon(np.expand_dims(computed_sinogram, axis=1), theta, center=None, algorithm=algorithm, sinogram_order=False)
    recon1 = np.squeeze(recon1)
    
    
    # Tomopy reconstruct partial noisy sinogram
    used_angles = mask.numpy()>0
    used_mask = np.expand_dims(mask.numpy()[used_angles], axis=1)
    recon2 = tomopy.recon(np.expand_dims(partial_noisy_sinogram[used_angles]/used_mask,axis=1),\
                          theta[used_angles], center=None, algorithm=algorithm, sinogram_order=False)
    recon2 = np.squeeze(recon2)
    
    # crop
    recon0 = crop(recon0,final_x,final_y)
    recon1 = crop(recon1,final_x,final_y)
    recon2 = crop(recon2,final_x,final_y)
    
    print('Predicted')
    MSE_predicted, SSIM_predicted, PSNR_predicted = compare(recon0, recon1)
    predicted_err = [MSE_predicted, SSIM_predicted, PSNR_predicted]
    
    print('Noisy')
    MSE_noisy, SSIM_noisy, PSNR_noisy = compare(recon0, recon2)
    noisy_err = [MSE_noisy, SSIM_noisy, PSNR_noisy]
    
    return(predicted_err, noisy_err, recon0, recon1, recon2)

def iradon_all(all_proj_samples, # sinograms
               all_masks,
               num_proj_pix,
               theta,
               algorithms,
               sqrt_reg,
               x_size,
               y_size,
               save_path,
               train = False,
               **kwargs,
               ):

    if train:
        mask = all_masks
        proj_sample = all_proj_samples
        
        mask_expand = tf.expand_dims(mask, axis=-1)
        mask_expand = tf.repeat(mask_expand, num_proj_pix, axis=-1)
        
    
        proj_sample_expand = tf.where(mask_expand>sqrt_reg, proj_sample/mask_expand, 
                                      proj_sample)
        
        all_input_encode = []
        for algorithm in algorithms:
            input_encode_f = tomopy.recon(proj_sample_expand, theta, center=None, sinogram_order=True, 
                                          algorithm=algorithm)
    
            input_encode_f = crop(input_encode_f,x_size,y_size, ignore_dim_0=True)
    
            # input_encode_f = iradon(proj_sample_expand, theta,
            #                         x_size, y_size,
            #                         filter_1d, 
            #                         )
            all_input_encode.append(input_encode_f)
    
        input_encode_f = tomopy.recon(mask_expand, theta, center=None, sinogram_order=True, 
                                      algorithm='fbp', filter_name='none')
        input_encode_f = crop(input_encode_f,x_size,y_size, ignore_dim_0=True)
        # input_encode_f = iradon(mask_expand, theta,
        #                         x_size, y_size,
        #                         filter_1d_vec[filter_mask_ind], 
        #                         )
    
        all_input_encode.append(input_encode_f)
        all_input_encode = tf.stack(all_input_encode,axis=-1)
        
        np.save(save_path + '/all_input_encode.npy', all_input_encode)
    else:
        all_input_encode = np.load(save_path + '/all_input_encode.npy')
        
    return(all_input_encode)

def toy_dist(mix_prob = 0.3,
             conc_0 = np.array([0.35580334, 0.94963009, 0.60227688, 0.43061459], dtype=np.float32),
             conc_1 = np.array([0.00390356, 0.44335424, 0.83152378, 0.52733124], dtype=np.float32),
             ):
    
    multivar_0 = tfd.Dirichlet(conc_0,
                               validate_args=False,
                               allow_nan_stats=True,
                               force_probs_to_zero_outside_support=False,
                               name='Dirichlet'
                               ) 

    multivar_1 = tfd.Dirichlet(conc_1,
                               validate_args=False,
                               allow_nan_stats=True,
                               force_probs_to_zero_outside_support=False,
                               name='Dirichlet'
                               ) 
    
    
    
    bimix_dist = tfd.Mixture(
      cat=tfd.Categorical(probs=[mix_prob, 1.-mix_prob]),
      components=[
        multivar_0,
        multivar_1,
    ])
    
    return(bimix_dist)
