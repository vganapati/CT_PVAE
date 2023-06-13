#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:12:35 2022

@author: vganapa1
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


def find_conv_output_dim(input_length, stride, kernel_size):
    # finds output dimension for 'valid' padding
    output_length = np.zeros([input_length,],dtype=np.int32)
    output_length[0::stride]=1
    output_length = output_length[:-kernel_size+1]
    output_length = np.sum(output_length)  
    return output_length


def create_encode_net(num_angles, # x dimension
                      num_proj_pix, # y dimension
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
                      feature_maps_multiplier,
                      reg,
                      encode_net_ind=0,
                      verbose=False,
                      x_size = None,
                      y_size = None,
                      num_filters = None,
                      ):
      
    num_feature_maps_vec = feature_maps_multiplier*np.array(num_feature_maps_vec)
    
    
    input_fbp = tf.keras.Input(name='input_fbp', 
                               shape=(x_size,y_size, num_filters+1), 
                               batch_size = batch_size)

    output = input_fbp


    output = tf.repeat(output, feature_maps_multiplier, axis=-1) # repeat to make divisible by feature_maps_multiplier
        

    if dropout_prob == 0:
        apply_dropout = False
    else:
        apply_dropout = True
    
    # Downsampling through the model
    skips_val = [output]
    skips_pixel_x = [output.shape[1]]
    skips_pixel_y = [output.shape[2]]
    skips_pixel_z = [output.shape[3]]


    for i in range(num_blocks):
        
        # intermediate layers, no residual connection
        # conv with stride = 1 and padding = same
        for l in range(intermediate_layers):
            output = \
                conv_block(output, # input
                           output.shape[-1], # output size channels
                           intermediate_kernel,
                           apply_norm = apply_norm, norm_type = norm_type,
                           apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                           initializer = initializer, 
                           transpose = False,
                           stride = (1,1), 
                           )   

        output = \
            conv_block(output, # input
                       num_feature_maps_vec[i], # output size channels
                       kernel_size,
                       apply_norm = apply_norm, norm_type = norm_type,
                       apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                       initializer = initializer, 
                       transpose = False,
                       stride = (stride_encode, stride_encode), 
                       )


        skips_val.append(output)    
        skips_pixel_x.append(output.shape[1])
        skips_pixel_y.append(output.shape[2])        
        skips_pixel_z.append(output.shape[3])

    model = keras.Model(inputs = (input_fbp), outputs = (skips_val), name='encode_net_' + str(encode_net_ind))

    if verbose:
        model.summary()
    
    return(model, skips_pixel_x, skips_pixel_y, skips_pixel_z)


def create_decode_net(skips_pixel_x,
                      skips_pixel_y,
                      skips_pixel_z,
                      batch_size,
                      final_output_channels, # number of output channels desired
                      kernel_size, 
                      stride_encode,
                      apply_norm, norm_type, 
                      initializer,
                      dropout_prob,
                      intermediate_layers,
                      intermediate_kernel,
                      feature_maps_multiplier, # should be the same value given to create_encode_net
                      decode_net_ind = 0,
                      verbose=False,
                      ):
    
    skips = []
    for skip in range(len(skips_pixel_x)):
        skip_input = tf.keras.Input(shape=(skips_pixel_x[skip],
                                           skips_pixel_y[skip],
                                           skips_pixel_z[skip]//feature_maps_multiplier), 
                                    batch_size = batch_size,
                                    name=str(skip))
        skips.append(skip_input)
    

    num_skips = len(skips)
           
    if dropout_prob == 0:
        apply_dropout = False
    else:
        apply_dropout = True    
    
    output = skips[-1]
    skips_reverse = reversed(skips[:-1])
    skips_pixel_z_reverse = reversed(skips_pixel_z[:-1])
    
    
    # Upsampling and establishing the skip connections
    for i, skip in enumerate(skips_reverse):
        
        output_channels = next(skips_pixel_z_reverse)
        
        output = conv_block(output, # input
                            output_channels, # output size channels
                            kernel_size, 
                            apply_norm = apply_norm, norm_type=norm_type,
                            apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                            initializer = initializer, 
                            transpose = True,
                            stride = (stride_encode, stride_encode),
                            )
        
        # intermediate layers, don't change shape
        # conv with stride = 1 and padding = same
        for l in range(intermediate_layers):
            output = \
                conv_block(output, # input
                           output.shape[-1], # output size channels
                           intermediate_kernel,
                           apply_norm = apply_norm, norm_type = norm_type,
                           apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
                           initializer = initializer, 
                           transpose = False,
                           stride = (1,1), 
                           )     


        skip_x = skip.shape[1]
        skip_y = skip.shape[2]
        
        # crop output
        remove_pad_x = output.shape[1] - skip_x
        remove_pad_y = output.shape[2] - skip_y
        
        r_x = remove_pad_x%2
        r_y = remove_pad_y%2
    
        output = output[:,remove_pad_x//2+r_x:remove_pad_x//2+r_x+skip_x,remove_pad_y//2+r_y:remove_pad_y//2+r_y+skip_y,:]
        if i<(num_skips-1): # do not concatenate input
            output = tf.keras.layers.Concatenate()([output, skip])

    # resize to the final output channel size
    output = \
    conv_block(output, # input
               final_output_channels*2, # output size channels
               kernel_size,
               apply_norm = apply_norm, norm_type = norm_type,
               apply_dropout=apply_dropout, dropout_prob = dropout_prob, 
               initializer = initializer, 
               transpose = False,
               stride = (1,1),
               )   

    output_mean, output_var = tf.split(output, [output.shape[-1]//2, output.shape[-1]//2], 
                                       axis=-1, num=None, name='split')

    model = keras.Model(inputs = (skips), outputs = (output_mean, output_var), name='decode_net_' + str(decode_net_ind))

    if verbose:
        model.summary()
        
    return(model)


def periodic_padding(image, padding_tuple): # padding is added to beginnings and ends of x and y
    '''
    Create a periodic padding (wrap) around the image, to emulate periodic boundary conditions
    https://github.com/tensorflow/tensorflow/issues/956
    
    usage example:

    image = tf.reshape(tf.range(30, dtype='float32'), shape=[5,6])
    padded_image = periodic_padding(image, padding=2)
    '''
    
    # XXX periodic boundary conditions coded here are not correct for sinograms

    padding_0,padding_1 = padding_tuple[0]
    partial_image = image
 
 
 
    if padding_0 != 0:  
        upper_pad = tf.repeat(image,int(np.ceil(padding_0/image.shape[1])),axis=1)[:,-padding_0:,:,:]
        # upper_pad = image[:,-padding_0:,:,:]
        partial_image = tf.concat([upper_pad, partial_image], axis=1)
        
    if padding_1 != 0: 
        lower_pad = tf.repeat(image,int(np.ceil(padding_1/image.shape[1])),axis=1)[:,:padding_1,:,:]
        # lower_pad = image[:,:padding_1,:,:]
        partial_image = tf.concat([partial_image, lower_pad], axis=1)


    padded_image = partial_image
    padding_0,padding_1 = padding_tuple[1]
  
    if padding_0 != 0:   
        left_pad = tf.repeat(partial_image,int(np.ceil(padding_0/image.shape[2])),axis=2)[:,:,-padding_0:,:]
        # left_pad = partial_image[:,:,-padding_0:,:]
        padded_image = tf.concat([left_pad, padded_image], axis=2)
        
    if padding_1 != 0:
        right_pad = tf.repeat(partial_image,int(np.ceil(padding_1/image.shape[2])),axis=2)[:,:,:padding_1,:]
        # right_pad = partial_image[:,:,:padding_1,:]
        padded_image = tf.concat([padded_image, right_pad], axis=2)
        
    return padded_image


def conv_block(x, # input
               output_last_dim, # output size channels
               kernel_size,
               apply_norm = False, norm_type='batchnorm',
               apply_dropout=False, dropout_prob = 0, 
               initializer = 'glorot_uniform', 
               transpose = False,
               stride = (2,2),
               use_bias = True,
               ):
    """
    Dropout => Conv2D => Maxout => Batchnorm

    """
    stride_x, stride_y = stride
    
    if apply_dropout:
        x = tf.keras.layers.Dropout(dropout_prob)(x)

    
    if transpose:
        x1 = keras.layers.Conv2DTranspose(output_last_dim, (kernel_size, kernel_size), 
                                          strides=stride, padding='same',
                                          dilation_rate=(1, 1), 
                                          use_bias=use_bias, 
                                          kernel_initializer=initializer,   
                                          output_padding=None,
                                          )(x)
        
        x2 = keras.layers.Conv2DTranspose(output_last_dim, (kernel_size, kernel_size), 
                                          strides=stride, padding='same',
                                          dilation_rate=(1, 1), 
                                          use_bias=use_bias, 
                                          kernel_initializer=initializer,   
                                          output_padding=None,
                                          )(x)        
        
    else:
        input_x = x.shape[-3]
        input_y = x.shape[-2]
        
        # Add padding such that the convolution doesn't change the input shape beyond stride effects

        if input_x%stride_x:
            pad_x = kernel_size - input_x%stride_x
        else: # no remainder
            pad_x = kernel_size - stride_x
        
        if input_y%stride_y:
            pad_y = kernel_size - input_y%stride_y
        else: # no remainder
            pad_y = kernel_size - stride_y
 
        r_x = pad_x%2
        r_y = pad_y%2


        x = periodic_padding(x,((pad_x//2+r_x,pad_x//2),(pad_y//2+r_y,pad_y//2)))


        x1 = keras.layers.Conv2D(output_last_dim, (kernel_size, kernel_size), strides=stride, padding='valid',
                                 kernel_initializer=initializer, use_bias=use_bias)(x)
        x2 = keras.layers.Conv2D(output_last_dim, (kernel_size, kernel_size), strides=stride, padding='valid',
                                       kernel_initializer=initializer, use_bias=use_bias)(x)

    
    ## Maxout
    x = tf.maximum(x1, x2)

    if apply_norm:
      if norm_type.lower() == 'batchnorm':
        x = keras.layers.BatchNormalization()(x)
      elif norm_type.lower() == 'instancenorm':
        x = InstanceNormalization()(x)

    return(x)


class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5, initializer = 'glorot_uniform'):
      super(InstanceNormalization, self).__init__()
      self.epsilon = epsilon
      self.initializer = initializer

    def build(self, input_shape):
      self.scale = self.add_weight(
          name='scale',
          shape=input_shape[-1:],
          initializer=self.initializer,
          trainable=True)

      self.offset = self.add_weight(
          name='offset',
          shape=input_shape[-1:],
          initializer='zeros',
          trainable=True)

    def call(self, x):
      mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
      inv = tf.math.rsqrt(variance + self.epsilon)
      normalized = (x - mean) * inv
      return self.scale * normalized + self.offset
      
