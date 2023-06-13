#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vganapa1, vdumont
"""

# System
import os
import sys
import time
import logging
import argparse

# Externals
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
# from skimage.transform.radon_transform import _get_fourier_filter

# Locals
from .create_masks import create_all_masks
from .forward_functions import project_tf_fast
from .helper_functions import create_folder, get_sinograms, create_tf_dataset, create_coords, find_loss_vae_unsup, plot, evaluate_sinogram, iradon_all, compare, plot_single_example
from .models import create_encode_net, create_decode_net

def get_args():
    ### Command line args ###
    parser = argparse.ArgumentParser(description='Get command line args')
    parser.add_argument('--ae', type=float, action='store', dest='adam_epsilon', 
                        help='adam_epsilon', default = 1e-7)
    parser.add_argument('-b', type=int, action='store', dest='batch_size',
                        help='batch size', default = 4)    
    parser.add_argument('--ns', type=int, action='store', dest='num_samples',
                        help='number of times to sample VAE in training', default = 2)  
    parser.add_argument('--det', action='store_true', dest='deterministic', 
                        help='no latent variable, simply maximizes log probability of output_dist') 
    parser.add_argument('--dp', type=float, action='store', dest='dropout_prob', 
                        help='dropout_prob, percentage of nodes that are dropped', default=0)
    parser.add_argument('--en', type=int, action='store', dest='example_num', 
                        help='example index for visualization', default = 0)
    parser.add_argument('-i', type=int, action='store', dest='num_iter', 
                        help='number of training iterations', default = 100)
    parser.add_argument('--ik', type=int, action='store', dest='intermediate_kernel',
                        help='intermediate_kernel for model_encode', default = 4)
    parser.add_argument('--il', type=int, action='store', dest='intermediate_layers', 
                        help='intermediate_layers for model_encode', default = 2)
    parser.add_argument('--input_path', action='store',
                        help='path to folder containing training data')
    parser.add_argument('--klaf', type=float, action='store', dest='kl_anneal_factor', 
                        help='multiply kl_anneal by this factor each iteration', default=1)
    parser.add_argument('--klm', type=float, action='store', dest='kl_multiplier', 
                        help='multiply the kl_divergence term in the loss function by this factor', default=1)
    parser.add_argument('--ks', type=int, action='store', dest='kernel_size',
                        help='kernel size in model_encode_I_m', default = 4)
    parser.add_argument('--lr', type=float, action='store', dest='learning_rate',
                        help='learning rate', default = 1e-4)
    parser.add_argument('--nb', type=int, action='store', dest='num_blocks', 
                        help='num convolution blocks in model_encode', default = 3)
    parser.add_argument('--nfm', type=int, action='store', dest='num_feature_maps', 
                        help='number of features in the first block of model_encode', default = 20)
    parser.add_argument('--nfmm', type=float, action='store', dest='num_feature_maps_multiplier', 
                        help='multiplier of features for each block of model_encode', default = 1.1)
    parser.add_argument('--norm', type=float, action='store', dest='norm', 
                        help='gradient clipping by norm', default=100)
    parser.add_argument('--normal', action='store_true', dest='use_normal', 
                        help='use a normal distribution as final distribution') 
    parser.add_argument('--nsa', type=int, action='store', dest='num_sparse_angles', \
                        help='number of angles to image per sample (dose remains the same)', default = 10)
    parser.add_argument('--api', type=int, action='store', dest='angles_per_iter', \
                        help='number of angles to check per iteration (stochastic optimization)', default = 5)
    parser.add_argument('--pnm', type=float, action='store', dest='poisson_noise_multiplier',
                        help='poisson noise multiplier, higher value means higher SNR', default = (2**16-1)*0.41)
    parser.add_argument('--pnm_start', type=float, action='store', dest='pnm_start', 
                        help='poisson noise multiplier starting value, anneals to pnm value', default = None)
    parser.add_argument('--train_pnm', action='store_true', dest='train_pnm', 
                        help='if True, make poisson_noise_multiplier a trainable variable')   
    parser.add_argument('-r', type=int, action='store', dest='restore_num', 
                        help='checkpoint number to restore from', default = None)
    parser.add_argument('--random', action='store_true', dest='random', 
                        help='if True, randomly pick angles for masks')
    parser.add_argument('--restore', action='store_true', dest='restore', \
                        help='restore from previous training')
    parser.add_argument('--save_path', action='store',
                        help='path to save output')
    parser.add_argument('--se', type=int, action='store', dest='stride_encode',
                        help='convolution stride in model_encode_I_m', default = 2)
    parser.add_argument('--si', type=int, action='store', dest='save_interval', 
                        help='save_interval for checkpoints and intermediate values', default = 100000)
    parser.add_argument('--td', type=int, action='store', dest='truncate_dataset', 
                        help='truncate_dataset by this value to not load in entire dataset; overriden when restoring a net', default = 100)
    parser.add_argument('--train', action='store_true', dest='train',
                        help='run the training loop')
    parser.add_argument('--ufs', action='store_true', dest='use_first_skip', 
                        help='use the first skip connection in the unet')
    parser.add_argument('--ulc', action='store_true', dest='use_latest_ckpt', \
                        help='uses latest checkpoint, overrides -r')
    parser.add_argument('--visualize', action='store_true', dest='visualize', 
                        help='visualize results')
    parser.add_argument('--pixel_dist', action='store_true', dest='pixel_dist', 
                        help='get distribution of each pixel in final reconstruction')
    parser.add_argument('--real', action='store_true', dest='real_data', 
                        help='denotes real data, does not simulate noise') 
    parser.add_argument('--no_pad', action='store_true', dest='no_pad', 
                        help='sinograms have no zero-padding') 
    parser.add_argument('--toy_masks', action='store_true', dest='toy_masks', 
                        help='uses the toy masks') 
    parser.add_argument('--algorithms', action='store', help='list of initial algorithms to use', 
                         nargs='+',default=['gridrec'])
    parser.add_argument('--no_final_eval', action='store_true', dest='no_final_eval', 
                        help='skips the final evaluation') 
    args = parser.parse_args()
    return args

class CT_VAE():
    def __init__(self,train=False,visualize=False,pixel_dist=False,input_path=None,
                 real_data=False,no_pad=False,truncate_dataset=100,
                 num_iter=100, poisson_noise_multiplier=(2**16-1)*0.41,pnm_start=None,
                 toy_masks=None,algorithms=['gridrec'],no_final_eval=False,**kwargs):
        
        """
        Initialization function
        
        Parameters
        ----------
        train : bool
            Run the training loop
        visualize : bool
            Visualize results
        """
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.setup_start_time = time.time()
        # Create folder for output
        create_folder(**kwargs)
        
        self.sqrt_reg = np.finfo(np.float32).eps.item() # regularizing sqrt in backprop
        self.no_pad = no_pad

        # annealing of poisson_noise_multiplier
        if pnm_start is not None:
            self.pnm_anneal_factor = np.exp(np.log(poisson_noise_multiplier/pnm_start)/num_iter)
        else:
            self.pnm_anneal_factor = 1.0

        # get sinograms
        self.x_train_sinograms, self.theta, self.num_proj_pix = get_sinograms(input_path)
        self.num_angles = len(self.theta)        
        
        ### Determine reconstruction x_size and y_size
        if self.no_pad:
            self.x_size = self.num_proj_pix
            self.y_size = self.num_proj_pix
        else:
            self.x_size = int(np.floor(self.num_proj_pix/np.sqrt(2)-2))
            self.y_size = int(np.floor(self.num_proj_pix/np.sqrt(2)-2))

        
        # self.x_size = int(np.load(input_path + '/x_size.npy'))
        # self.y_size = int(np.load(input_path + '/y_size.npy'))

        # mask and add noise to the sinograms
        self.all_masks, self.all_proj_samples = create_all_masks(self.x_train_sinograms, 
                                                                 self.num_angles,
                                                                 real_data = real_data,
                                                                 poisson_noise_multiplier = poisson_noise_multiplier,
                                                                 train = train,
                                                                 truncate_dataset = truncate_dataset,
                                                                 toy_masks = toy_masks,
                                                                 **kwargs)

        # perform filtered back-projection on the noisy partial sinograms
        
        # filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None]
        # filter_mask_ind = -1
        # self.filter_1d_vec = []
        # for filter_name in filters:
        #     filter_1d = np.squeeze(_get_fourier_filter(self.num_proj_pix, filter_name), axis=-1)
        #     # plt.plot(filter_1d)
        #     self.filter_1d_vec.append(filter_1d)
        # # self.filter_1d_vec = np.concatenate(self.filter_1d_vec,axis=-1)
        
        # filters = ['shepp', 'cosine', 'hamming', 'hann', 'ramlak', 'parzen', 'butterworth', 'none']
        # filters = ['none']
        # self.filter_1d_vec = filters
        # filter_mask_ind = -1
        
        
        self.algorithms = algorithms
        self.num_algorithms = len(algorithms)

        
        self.all_input_encode = iradon_all(self.all_proj_samples, # sinograms
                                           self.all_masks,
                                           self.num_proj_pix,
                                           self.theta,
                                           self.algorithms,
                                           self.sqrt_reg,
                                           self.x_size,
                                           self.y_size,
                                           train = train,
                                           **kwargs
                                           )
        # print(self.all_input_encode.shape)
        self.create_dataset(input_path=input_path, truncate_dataset=truncate_dataset,
                            **kwargs)
        self.create_networks(**kwargs)
        self.get_started(poisson_noise_multiplier=poisson_noise_multiplier,
                         **kwargs)
        if train:
            self.train(num_iter=num_iter,
                       **kwargs)
        if no_final_eval:
            pass
        else:
            self.final_evaluation(**kwargs)
        if visualize:
            self.visualization(input_path=input_path,
                               real_data=real_data,
                               **kwargs)
        if pixel_dist:
            self.pixel_dist(input_path=input_path,
                            **kwargs)
            
    def create_dataset(self,input_path=None,save_path=None,truncate_dataset=100,batch_size=4, angles_per_iter=5,**kwargs):

        """
        Build dataset from generated data
        
        Parameters
        ----------
        input_path : str
            Path to folder containing training data
        truncate_dataset : int
            Truncate_dataset by this value to not load in entire dataset; overriden when restoring a net
        batch_size : int
            Data batch size
        """
        ### Set parameters
        buffer_size = 100 # shuffle buffer

        self.train_ds, self.train_size, self.train_ds_unshuffled, self.train_ds_angles = create_tf_dataset(self.x_train_sinograms,
                                                                                                           self.all_proj_samples,
                                                                                                           self.all_masks,
                                                                                                           self.all_input_encode,
                                                                                                           truncate_dataset,
                                                                                                           batch_size,buffer_size,
                                                                                                           self.num_angles,
                                                                                                           self.theta,
                                                                                                           angles_per_iter,
                                                                                                           **kwargs
                                                                                                           )

    def create_networks(self,batch_size=4,num_feature_maps=20,num_feature_maps_multiplier=1.1,num_blocks=3,kernel_size=4,stride_encode=2,dropout_prob=0,
                        intermediate_layers=2,intermediate_kernel=4,deterministic=False,**kwargs):
        """
        Build encoder and decoder neural networks
        
        Parameters
        ----------
        batch_size : int
            Data batch size
        num_feature_maps : int
             Number of features in the first block of model_encode
        num_feature_maps_multiplier : float
            Multiplier of features for each block of model_encode
        num_blocks : int
            Number of convolution blocks in model_encode
        kernel_size : int
            Kernel size in model_encode_I_m
        stride_encode : int
            Convolution stride in model_encode_I_m
        dropout_prob : float
            Percentage of nodes that are dropped
        intermediate_layers : int
            Intermediate layers for model_encode
        intermediate_kernel : int
            Intermediate kernel for model_encode
        deterministic : bool
            No latent variable, simply maximizes log probability of output_dist
        """
        apply_norm = False # If True, use batch normalization
        norm_type = 'batchnorm' #'batchnorm' or 'instancenorm'
        # Initializers in: https://www.tensorflow.org/api_docs/python/tf/keras/initializers
        initializer = tf.keras.initializers.GlorotUniform()
        # initializer = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        
        # Neural Networks
        # Coordinates, angles x spatial coords
        coords = create_coords(self.num_angles,self.num_proj_pix,batch_size)
        num_feature_maps_vec = [int(num_feature_maps*num_feature_maps_multiplier**i) for i in range(num_blocks)]
        if deterministic:
            self.feature_maps_multiplier = 1
        else:
            self.feature_maps_multiplier = 2


        self.model_encode, self.skips_pixel_x, self.skips_pixel_y, self.skips_pixel_z = \
        create_encode_net(self.num_angles, # x dimension
                          self.num_proj_pix, # y dimension
                          num_feature_maps_vec,
                          batch_size,
                          num_blocks, 
                          kernel_size, 
                          stride_encode,
                          apply_norm, norm_type, 
                          initializer,
                          dropout_prob,
                          intermediate_layers,
                          intermediate_kernel,
                          coords,
                          self.feature_maps_multiplier,
                          self.sqrt_reg,
                          x_size = self.x_size,
                          y_size = self.y_size,
                          num_filters = self.num_algorithms,
                          verbose=False,
                          )

        self.model_decode = \
        create_decode_net(self.skips_pixel_x,
                          self.skips_pixel_y,
                          self.skips_pixel_z,
                          batch_size,
                          1, # number of output channels desired
                          kernel_size, 
                          stride_encode,
                          apply_norm, norm_type, 
                          initializer,
                          dropout_prob,
                          intermediate_layers,
                          intermediate_kernel,
                          self.feature_maps_multiplier, # should be the same value given to create_encode_net
                          verbose=False,
                          )
        logging.info('-'*40)
        ngen = np.sum([np.prod(v.get_shape()) for v in self.model_encode.trainable_weights])
        ndis = np.sum([np.prod(v.get_shape()) for v in self.model_decode.trainable_weights])
        logging.info('{:,d} parameters in encoder'.format(ngen))
        logging.info('{:,d} parameters in decoder'.format(ndis))
        logging.info('Number of parameters : {:,d}'.format(ngen+ndis))
        logging.info('-'*40)

    def get_started(self,learning_rate=1e-4,adam_epsilon=1e-7,save_path=None,restore=False,restore_num=None,
                    use_latest_ckpt=False,batch_size=4,use_normal=False,num_blocks=3,poisson_noise_multiplier=(2**16-1)*0.41,**kwargs):

        self.kl_anneal = tf.Variable(1, dtype=tf.float32)
        self.poisson_noise_multiplier = tf.Variable(poisson_noise_multiplier, dtype=tf.float32)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=adam_epsilon)
        # save checkpoints
        checkpoint_dir = 'training_checkpoints'
        self.checkpoint_prefix = os.path.join(save_path, checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(model_encode=self.model_encode, model_decode=self.model_decode, 
                                              optimizer=self.optimizer, kl_anneal=self.kl_anneal,
                                              poisson_noise_multiplier = self.poisson_noise_multiplier)    
        if restore:
            # restore a checkpoint
            if use_latest_ckpt:
                self.checkpoint.restore(tf.train.latest_checkpoint(os.path.join(save_path, checkpoint_dir)))
            else:
                self.checkpoint.restore(os.path.join(save_path, checkpoint_dir,'ckpt-')+str(restore_num))
        # prior on z, the latent variable 
        skip_shapes = np.array([batch_size*np.ones_like(self.skips_pixel_x), self.skips_pixel_x, self.skips_pixel_y,
                                np.array(self.skips_pixel_z)//self.feature_maps_multiplier]).T
        if use_normal:
            self.prior = [tfd.Normal(loc=tf.zeros(skip_shapes[i]), scale=1) for i in range(num_blocks+1)]
        else:
            self.prior = [tfd.Beta(0.5*tf.ones(skip_shapes[i]), 0.5*tf.ones(skip_shapes[i])) for i in range(num_blocks+1)]
        self.trainable_vars = self.model_encode.trainable_variables + self.model_decode.trainable_variables
    
    def train(self,num_iter=100,kl_anneal_factor=1,save_interval=10000,save_path=None,**kwargs):
        train_loss_vec = []
        train_loss_kl = []
        train_loss_loglik = []
        iter_vec = []
        
        if num_iter == 0:
            start_time = time.time()
        ### Training and Validation ###
        for iter_i in range(num_iter):
            # print(tf.config.experimental.get_memory_usage("GPU:0"))
            self.kl_anneal.assign(tf.minimum(tf.maximum(self.kl_anneal*kl_anneal_factor,0),100))
            # print(self.kl_anneal)
            proj, proj_sample, mask, input_encode = next(self.train_ds)
            angles_i = next(self.train_ds_angles)
            # print(angles_i)
            loss_M_VAE, _, _, _, kl_divergence, loglik, _ = self.train_step(proj_sample,mask,input_encode,True,
                                                           self.poisson_noise_multiplier*self.pnm_anneal_factor**iter_i,
                                                           angles_i,no_pad=self.no_pad,**kwargs)
            logging.info('Epoch {:>3} / {:<3} {:>12} {:>11.5f}'.format(iter_i+1,num_iter,'loss :',loss_M_VAE))
            print('Iteration number: ' + str(iter_i))
            print('Training loss_M_VAE: ' + str(loss_M_VAE))
            train_loss_vec.append(loss_M_VAE)
            train_loss_kl.append(kl_divergence)
            train_loss_loglik.append(loglik)
            
            if np.isnan(loss_M_VAE):
                sys.exit()
            if iter_i == 0:
                setup_end_time = time.time()
                setup_time = (setup_end_time-self.setup_start_time)/60 # minutes
                np.save(save_path + '/setup_time.npy', setup_time)
                # print('Setup took ' + str((setup_end_time-self.setup_start_time)/60) + ' minutes.')
                start_time = time.time()
            if((iter_i%save_interval == 0) or (iter_i == num_iter-1)):      
                iter_vec.append(iter_i)
                np.save(save_path + '/train_loss_vec.npy', train_loss_vec)
                np.save(save_path + '/train_loss_kl.npy', train_loss_kl)
                np.save(save_path + '/train_loss_loglik.npy', train_loss_loglik)
                np.save(save_path + '/iter_vec.npy', iter_vec)
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        end_time = time.time()
        training_time = (end_time-start_time)/60
        np.save(save_path + '/training_time.npy', training_time)
        # print('Training took ' + str((end_time-start_time)/60) + ' minutes.')
        plot(save_path, train_loss_vec, 'Training loss', 'train_loss_vec')
        plot(save_path, train_loss_kl, 'Training loss KL divergence', 'train_loss_kl')
        plot(save_path, train_loss_loglik, 'Training loss loglikelihood', 'train_loss_loglik')
        
        # print('poisson_noise_multiplier')
        # print(self.poisson_noise_multiplier)
    
    def final_evaluation(self,batch_size=4,save_path=None,**kwargs):
        ds = self.train_ds_unshuffled
        # print('Starting final evaluation...')
        start_time = time.time()
        loss_final = []
        # sinogram_final = []
        reconstruction_final = []
        
        for ind in range(self.train_size//batch_size):

            proj, proj_sample, mask, input_encode = next(ds)
            loss_M_VAE, output_dist_vec, q, q_sample, kl_divergence, loglik, input_encode = self.train_step(proj_sample, mask, input_encode, 
                                                                                                            False, self.poisson_noise_multiplier,
                                                                                                            batch_size=batch_size, 
                                                                                                            no_pad=self.no_pad,**kwargs)
            # print('loss:')
            # print(loss_M_VAE)
            loss_final.append(loss_M_VAE)
            # sinogram_final.append(np.squeeze(output_dist_vec[0].sample()[0],axis=-1)) # mean
            reconstruction_final.append(output_dist_vec[0].sample()) # mean
                
        loss_final = np.stack(loss_final)
        # sinogram_final = np.concatenate(sinogram_final,axis=0)
        reconstruction_final = np.concatenate(reconstruction_final,axis=0)
        np.save(save_path + '/loss_final.npy', loss_final)
        np.save(save_path + '/reconstruction_final.npy', reconstruction_final)
        self.loss_final_mean = np.mean(loss_final)
        logging.info('Average loss final : {}'.format(self.loss_final_mean))
        # print('Average loss final:')
        # print(self.loss_final_mean)
        end_time = time.time()
        
        final_train_time = (end_time-start_time)/60
        np.save(save_path + '/final_train_time.npy', final_train_time)
        # print('Final train took ' + str((end_time-start_time)/60) + ' minutes.')

    @tf.function
    def train_step(self,proj_sample,mask,input_encode,training,poisson_noise_multiplier_i,angles_i=None,batch_size=4,deterministic=False,num_blocks=3,
                   use_normal=False,kl_multiplier=1,norm=100,
                   num_samples=2, train_pnm=False, **kwargs):
        if train_pnm:
            all_trainable_vars = self.trainable_vars + [self.poisson_noise_multiplier]
        else:
            all_trainable_vars = self.trainable_vars
        with tf.GradientTape(watch_accessed_variables=True, persistent=False) as tape:
            tape.watch(all_trainable_vars)
            loss_M_VAE, output_dist_vec, q, q_sample, kl_divergence, loglik, input_encode = \
                find_loss_vae_unsup(proj_sample,mask,input_encode,self.num_proj_pix,self.x_size,self.y_size,deterministic,num_blocks,self.model_encode,
                                    self.model_decode,poisson_noise_multiplier_i,self.sqrt_reg,batch_size,self.prior,use_normal,training,
                                    self.kl_anneal,kl_multiplier,theta = self.theta, angles_i=angles_i, pad=not(self.no_pad), 
                                    num_samples=num_samples)
            loss_M_VAE = tf.reduce_mean(loss_M_VAE)/1e5
        if training:
            # loss_M_VAE
            gradients = tape.gradient(loss_M_VAE, all_trainable_vars)
            gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in gradients]
            gradients = [tf.clip_by_norm(g, norm)
                          for g in gradients]   
            self.optimizer.apply_gradients(zip(gradients, all_trainable_vars))
        return(loss_M_VAE, output_dist_vec, q, q_sample, kl_divergence, loglik, input_encode)

    def visualization(self,example_num=0,input_path=None,save_path=None,real_data=False,
                      num_sparse_angles=10,**kwargs):
        # print(num_sparse_angles)
        proj,mask,proj_sample, input_encode = self.load_batch(example_num,**kwargs)
        # input_encode is the filtered back-propagation results
        loss_M_VAE, output_dist_vec, q, q_sample, kl_divergence, loglik, input_encode = self.train_step(proj_sample,mask,input_encode,False,
                                                                                                        self.poisson_noise_multiplier,
                                                                                                        no_pad=self.no_pad,
                                                                                                        **kwargs)
        print('loss_M_VAE')
        print(loss_M_VAE)
        
        print('kl_divergence')
        print(kl_divergence)
        
        print('loglik')
        print(loglik)
        
        try:
            train_loss_vec = np.load(save_path + '/train_loss_vec.npy')
            plot(save_path, train_loss_vec, 'Training loss', 'train_loss_vec')
            
            train_loss_kl = np.load(save_path + '/train_loss_kl.npy')
            plot(save_path, train_loss_kl, 'Training loss KL divergence', 'train_loss_kl')
            
            train_loss_loglik = np.load(save_path + '/train_loss_loglik.npy')
            plot(save_path, train_loss_loglik, 'Training loss loglikelihood', 'train_loss_loglik')
            
        except FileNotFoundError:
            pass
        actual_sinogram = self.x_train_sinograms[example_num]
        actual_sinogram_dist = tfd.Poisson(actual_sinogram*self.poisson_noise_multiplier)
        actual_sinogram_noisy = actual_sinogram_dist.sample()/self.poisson_noise_multiplier  

        output_reconstruction = np.squeeze(output_dist_vec[0].sample()[0],axis=-1) # mean() or sample()
        computed_sinogram = tf.squeeze(project_tf_fast(output_reconstruction,self.theta,pad = not(self.no_pad), dim=2,
                                                       integrate_vae=False),axis=-1)

        partial_noisy_sinogram = proj_sample[0].numpy()
        algorithm = 'gridrec' # 'sirt', 'tv', 'gridrec', 'fbp'
        final_x = int(self.x_size) # actual size of reconstruction (tomopy reconstructs blank space)
        final_y = int(self.y_size) # actual size of reconstruction (tomopy reconstructs blank space)
        mask = mask[0]
        # print('mask')
        # print(mask)
        predicted_err, noisy_err, recon0, recon1, recon2 = evaluate_sinogram(actual_sinogram_noisy, computed_sinogram, partial_noisy_sinogram, 
                                                                             mask, self.theta, final_x=final_x, final_y=final_y, algorithm=algorithm)
        # plots
        plt.figure()
        plt.title('Actual Sinogram')
        plt.imshow(actual_sinogram_noisy,cmap='gray')
        vmin = np.min(actual_sinogram_noisy)
        vmax = np.max(actual_sinogram_noisy)
        plt.colorbar()
        plt.savefig(save_path + '/ActualSinogramNoisy.png')
        # print('Actual Sinogram Noisy')
        # print(actual_sinogram_noisy)
        
        plt.figure()
        plt.title('Computed Sinogram')
        plt.imshow(computed_sinogram,vmin=vmin, vmax=vmax,cmap='gray')
        plt.colorbar()
        plt.savefig(save_path + '/ComputedSinogram.png')

        
        plt.figure()
        plt.title('Input Partial Sinogram')
        plt.imshow(partial_noisy_sinogram, vmin=vmin, vmax=vmax/num_sparse_angles,cmap='gray')
        plt.colorbar()
        plt.savefig(save_path + '/InputPartialSinogram.png')
        plt.figure()
        # print('Input Partial Sinogram')
        # print(partial_noisy_sinogram)
        
        if real_data:
            vmin_1 = None
            vmax_1 = None
        else:
            prefix = input_path[8:]
            ground_truth = np.load(prefix + '_training.npy')[example_num]
            plt.title('Ground Truth')
            plt.imshow(ground_truth,cmap='gray')
            vmin_1 = np.min(ground_truth)
            vmax_1 = np.max(ground_truth)
            plt.colorbar()
            plt.savefig(save_path + '/GroundTruth.png')
            plt.figure()
            # print('Ground Truth')
            # print(ground_truth)
            
            plot_single_example(ground_truth, 'ground_truth', 
                                save_path,
                                example_num,
                                vmin=vmin_1, vmax=vmax_1)
            
        plt.title('Recon from Actual Sinogram')
        plt.imshow(recon0,cmap='gray', vmin=vmin_1, vmax=vmax_1)
        plt.colorbar()
        plt.savefig(save_path + '/ReconFromActualSinogram.png')

        plot_single_example(recon0, 'recon_actual_sinogram', 
                            save_path,
                            example_num,
                            vmin=vmin_1, vmax=vmax_1)
            
        plt.figure()
        plt.title('Recon from P-VAE')
        plt.imshow(output_reconstruction, vmin=vmin_1, vmax=vmax_1,cmap='gray')
        plt.colorbar()
        plt.savefig(save_path + '/ReconFromPVAE.png')
        print('Recon from P-VAE')
        print(output_reconstruction)

        plot_single_example(output_reconstruction, 'recon_PVAE', 
                            save_path,
                            example_num,
                            vmin=vmin_1, vmax=vmax_1)
            
        plt.figure()
        plt.title('Recon from Input Partial, Noisy Sinogram')
        plt.imshow(recon2, vmin=vmin_1, vmax=vmax_1,cmap='gray')
        plt.colorbar()
        plt.savefig(save_path + '/ReconFromInputPartialSinogram.png')

        plot_single_example(recon2, 'recon_partial_sino', 
                            save_path,
                            example_num,
                            vmin=vmin_1, vmax=vmax_1)
        
        recon0 = np.minimum(recon0,1)
        recon0 = np.maximum(recon0,0)
        print('Reconstruction from full sinogram:')
        compare(ground_truth, recon0, verbose=True)

        recon2 = np.minimum(recon2,1)
        recon2 = np.maximum(recon2,0)        
        print('Reconstruction from partial noisy sinogram')
        compare(ground_truth, recon2, verbose=True)
        
        print('Reconstruction from P-VAE')
        compare(ground_truth, output_reconstruction, verbose=True)
        
        np.save('ground_truth.npy',ground_truth)
        np.save('recon0.npy',recon0)
        np.save('recon2.npy',recon2)
        np.save('output_reconstruction.npy',output_reconstruction)
        
    def load_batch(self,ind,batch_size=4,**kwargs):
        proj = self.x_train_sinograms[ind]
        mask = self.all_masks[ind]
        proj_sample = self.all_proj_samples[ind]
        input_encode = self.all_input_encode[ind]
        
        proj = tf.repeat(tf.expand_dims(proj,axis=0), batch_size, axis=0)
        mask = tf.repeat(tf.expand_dims(mask,axis=0), batch_size, axis=0)
        proj_sample = tf.repeat(tf.expand_dims(proj_sample,axis=0), batch_size, axis=0)
        input_encode = tf.repeat(tf.expand_dims(input_encode,axis=0), batch_size, axis=0)
        
        return(proj,mask,proj_sample,input_encode)

    def pixel_dist(self,example_num=0,input_path=None,save_path=None,num_repeats=10000,num_samples=2,
                   num_samples_1=100,**kwargs):

        proj,mask,proj_sample, input_encode = self.load_batch(example_num,**kwargs)
        
        prefix = input_path[8:]
        ground_truth = np.load(prefix + '_training.npy')[example_num]
        ground_truth_reshape = tf.reshape(ground_truth,[-1])
        
        try:
            all_reconstructions = np.load(save_path + '/all_reconstructions_' + str(example_num) +'.npy')
        except FileNotFoundError:
            all_reconstructions = []
            for r in range(num_repeats):
                # input_encode is the filtered back-propagation results
                loss_M_VAE, output_dist_vec, q, q_sample, kl_divergence, loglik, input_encode = self.train_step(proj_sample,mask,input_encode,False,
                                                                                                                self.poisson_noise_multiplier,
                                                                                                                no_pad=self.no_pad, num_samples=num_samples,
                                                                                                                **kwargs)
    
                for s in range(num_samples):
                    # tf.squeeze(output_dist_vec[s].sample(num_samples_1)[0],axis=-1)
                    output_reconstruction = tf.squeeze(output_dist_vec[s].sample(num_samples_1),axis=-1) # mean()
                    (x0,x1,x2,x3) = np.shape(output_reconstruction)
                    output_reconstruction = tf.reshape(output_reconstruction,[x0*x1,x2*x3])
                    # print(output_reconstruction.shape)
                    all_reconstructions.append(output_reconstruction)
                    
            all_reconstructions = tf.concat(all_reconstructions,axis=0)
            np.save(save_path + '/all_reconstructions_' + str(example_num) +'.npy', all_reconstructions)
        # print(all_reconstructions.shape)
        
        # True distribution for discrete case
        
      
        x_train_0 = np.array([[1,2],[3,4]])/10
        x_train_0_reshape = tf.reshape(x_train_0,[-1])
        
        x_train_1 = np.array([[3,4],[1,2]])/10
        x_train_1_reshape = tf.reshape(x_train_1,[-1])
        
        # P(M | O_0)
        proj_0 = self.x_train_sinograms[2]
        proj_masked_0 = proj_0*tf.expand_dims(self.all_masks[example_num], axis=1)
        proj_dist_0 = tfd.Poisson(proj_masked_0*self.poisson_noise_multiplier)
        likelihood_0 = tf.reduce_sum(proj_dist_0.log_prob(proj_sample*self.poisson_noise_multiplier))

        # P(M | O_1)
        proj_1 = self.x_train_sinograms[0]
        proj_masked_1 = proj_1*tf.expand_dims(self.all_masks[example_num], axis=1)
        proj_dist_1 = tfd.Poisson(proj_masked_1*self.poisson_noise_multiplier)                                     
        likelihood_1 = tf.reduce_sum(proj_dist_1.log_prob(proj_sample*self.poisson_noise_multiplier))

        # P(O_0 | M)
        h_0 = likelihood_0 / (likelihood_0 + likelihood_1)
        
        # P(O_1 | M)
        h_1 = likelihood_1 / (likelihood_0 + likelihood_1)

        # print([x_train_0_reshape[pixel_ind].numpy(), x_train_1_reshape[pixel_ind].numpy()])
        # print([h_0.numpy(), h_1.numpy()])        
        
        for pixel_ind in range(all_reconstructions.shape[1]):
            plt.figure()
            delta_bin = 0.01
            bins = np.arange(5*delta_bin/10,0.5+delta_bin,delta_bin)
            # plt.title('True value: ' + str(ground_truth_reshape[pixel_ind]))
            (n, bins, _) = plt.hist(all_reconstructions[:,pixel_ind], bins=bins, histtype="stepfilled", density=True)
            
            plt.figure(figsize=[10, 5])
            plt.bar((bins[:-1]+bins[1:])/2, n/np.sum(n), width=0.01,label='P-VAE')
            # plt.plot(ground_truth_reshape[pixel_ind]*np.ones(50),np.arange(50))
            plt.xlim([0,0.5])
            plt.ylim([0, 1.0])
            plt.savefig(save_path + '/pixel_vae_' + str(pixel_ind) + '_example' + str(example_num) +'.png') 

            # plt.figure(figsize=[12.5, 4])
            # plt.title('True value: ' + str(ground_truth_reshape[pixel_ind]))
            plt.bar([x_train_0_reshape[pixel_ind], x_train_1_reshape[pixel_ind]], [h_0, h_1], 
                    width=delta_bin, edgecolor='black', alpha=0.2, hatch='//',label='True Posterior')
            plt.xlim([0,0.5])
            plt.savefig(save_path + '/pixel_vae_true_exp_' + str(pixel_ind) + '_ex_' + str(example_num) + '_no_leg.png', dpi=300)         
            plt.legend()
            plt.savefig(save_path + '/pixel_vae_true_exp_' + str(pixel_ind) + '_ex_' + str(example_num) + '.png', dpi=300)         
        
def main(**kwargs):
    vae = CT_VAE(**kwargs)
    return vae.loss_final_mean
