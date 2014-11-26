# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:18:23 2014

@author: vivianapetrescu
"""
import numpy as np
import time

import scipy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.signal import conv

class LeNetLayerConvPoolSeparableNonSymbolic:
    def __init__(self, rng):
        self.rng = rng
    def run_batch(self, input_images, image_shape, filter_shape,  Pstruct, b, poolsize):
        assert image_shape[1] == filter_shape[1]
        # the bias is a 1D tensor -- one bias per output feature map
        # convolve input feature maps with filters
        batch_size = image_shape[0]             
        fwidth = Pstruct[0]['U1'].shape[0]
        fheight = Pstruct[0]['U2'].shape[0]
        self.nbr_channels = image_shape[1]
        nbr_filters = Pstruct[0]['U3'].shape[0]
        initial_n_rows = image_shape[2]
        initial_n_cols = image_shape[3]
        
        # Final number of rows and columns        
        final_n_rows = initial_n_rows - fwidth + 1
        final_n_cols = initial_n_cols - fheight + 1
        # The convolved input images
        self.input4D = np.zeros((batch_size, nbr_filters, final_n_rows, final_n_cols))
        print 'batch size ', batch_size        
        one_image_shape = (self.nbr_channels, initial_n_rows, initial_n_cols)
       # assert one_image_shape == (1,28,28)
        for image_index in range(batch_size):
                # Convolve image with index image_index in the batch
                self.input4D = self.convolve_one_image(input_images[image_index,:,:,:],
                              one_image_shape,
                              Pstruct, 
                              filter_shape, 
                              image_index)   
    
   #     print 'before downsample', self.input4D
        # downsample each feature map individually, using maxpooling
        start = time.time()
        pooled_out = downsample.max_pool_2d(input=self.input4D,
                                            ds=poolsize, 
                                            ignore_border=True)
        end = time.time()
        self.downsample_time = (end - start)*1000/ image_shape[0]
                
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        sb = theano.shared(b)        
        self.output = T.tanh(pooled_out + sb.dimshuffle('x', 0, 'x', 'x'))
        return self.output.eval()
    
    """TODO change to have an image such as nbr channels as well"""
    def convolve_one_image(self, one_image, img_shape, 
                           Pstruct, filter_shape,
                           image_index):
   
  #      print Pstruct[0]['U1'].shape
  #      print Pstruct[0]['U2'].shape
  #      print Pstruct[0]['U3'].shape
        rank = Pstruct[0]['U1'].shape[1]
        fwidth = Pstruct[0]['U1'].shape[0]
        fheight = Pstruct[0]['U2'].shape[0]
        #
        #    #   rank 4, w,h 3x3, nbr filter 7
        num_input_feature_maps = img_shape[0]
            
        n_rows = img_shape[1] - fwidth + 1
        n_cols = img_shape[2] - fheight + 1
        horizontal_conv_out = np.zeros((rank, num_input_feature_maps, img_shape[1], n_cols))
        vertical_conv_out = np.zeros((rank, num_input_feature_maps, n_rows, n_cols))
        for chanel in range(num_input_feature_maps):        
                horizontal_filter_shape = (rank, 1, fwidth)
                horizontal_filters = np.ndarray(horizontal_filter_shape)
                horizontal_filters[:, 0, :] = np.transpose(Pstruct[chanel]['U1']);
                for r in range(rank):
                      horizontal_conv_out[r,chanel,:,:] = scipy.signal.convolve2d(one_image[chanel,:,:], 
                                                              horizontal_filters[r,:,:], mode='valid')
    #    
                vertical_filter_shape = (rank, fheight,1)
                vertical_filters = np.ndarray(vertical_filter_shape)        
                vertical_filters[:,:, 0] = np.transpose(Pstruct[chanel]['U2']);
                for r in range(rank):            
                  vertical_conv_out[r,chanel,:,:] = scipy.signal.convolve2d(horizontal_conv_out[r,chanel,:,:], 
                                                    vertical_filters[r,:,:], mode='valid')
    ##        ## numberof images, number of filters, image width, image height
        nbr_filters = Pstruct[0]['U3'].shape[0]
        for filter_index in range(nbr_filters):            
            output_image = np.zeros((num_input_feature_maps,n_rows, n_cols))
            for chanel in range(num_input_feature_maps):
                temp = np.zeros((n_rows, n_cols))
                alphas = Pstruct[chanel]['U3']
                f = filter_index
                for r in range(rank):
                         out = vertical_conv_out[r,chanel, :,:]* alphas[f, r] * Pstruct[chanel]['lmbda'][r]; 
                         temp = temp + out
                output_image[chanel, :, :] = temp
            output_image = np.sum(output_image, axis=0)
            self.input4D[image_index,filter_index,:,:] =  output_image
         
        return self.input4D