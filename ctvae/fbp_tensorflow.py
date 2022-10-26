#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:38:26 2022

@author: vganapa1
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def iradon(sinogram, theta,
           x_size, y_size,
           filter_1d, 
           ):
    
    """Inverse radon transform.
    Reconstruct an image from the radon transform, using the filtered
    back projection algorithm.

    sinogram is batch_size x num_angles x num_proj_pix
    
    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989
    Notes
    -----
    It applies the Fourier slice theorem to reconstruct an image by
    multiplying the frequency domain of the filter with the FFT of the
    projection data. This algorithm is called filtered back projection.
    """

    # batch_size = sinogram.shape[0]
    num_angles = len(theta)
    num_proj_pix = sinogram.shape[2]
    
    if num_angles != sinogram.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    
    # Apply filter in Fourier domain
    projection = tf.signal.fft(tf.cast(sinogram, tf.complex128)) * filter_1d
    radon_filtered = tf.math.real(tf.signal.ifft(projection))

    coords_x = tf.range(0,x_size, dtype=tf.float64) - x_size/2
    coords_y = tf.range(0,y_size, dtype=tf.float64) - y_size/2
    xpr, ypr = tf.meshgrid(coords_x,coords_y, indexing='ij')

    xpr = tf.repeat(tf.expand_dims(xpr,axis=-1), num_angles, axis=-1)
    ypr = tf.repeat(tf.expand_dims(ypr,axis=-1), num_angles, axis=-1)
    
    t = ypr * tf.math.cos(theta) - xpr * tf.math.sin(theta)

    coords_radon = tf.range(0,num_proj_pix, dtype=tf.float64)-num_proj_pix/2
    
    all_fbp = []    
    # XXX Vectorize
    for angle_ind in range(num_angles):
        fbp = tfp.math.interp_regular_1d_grid(x=t[:,:,angle_ind], 
                                              x_ref_min=tf.reduce_min(coords_radon), 
                                              x_ref_max=tf.reduce_max(coords_radon), 
                                              y_ref=radon_filtered[:,angle_ind,:],
                                              axis=-1)
        all_fbp.append(fbp)
    reconstructed = tf.reduce_sum(tf.stack(all_fbp,-1),-1)

    reconstructed = reconstructed * np.pi / (2 * num_angles)
    return(reconstructed)