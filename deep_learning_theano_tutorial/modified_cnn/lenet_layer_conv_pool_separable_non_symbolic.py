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
        self.nbr_filters = Pstruct[0]['U3'].shape[0]
        initial_n_rows = image_shape[2]
        initial_n_cols = image_shape[3]
        
        # Final number of rows and columns        
        final_n_rows = initial_n_rows - fwidth + 1
        final_n_cols = initial_n_cols - fheight + 1
        # The convolved input images
        self.input4D = np.zeros((batch_size, self.nbr_filters, final_n_rows, final_n_cols))
        one_image_shape = (self.nbr_channels, initial_n_rows, initial_n_cols)
       # assert one_image_shape == (1,28,28)
        nbr_filters = Pstruct[0]['U3'].shape[0]
        rank = Pstruct[0]['U1'].shape[1]
        pcoef = np.ndarray((nbr_filters, self.nbr_channels, rank))
        for filter_index in xrange(nbr_filters):
            for chanel in xrange(self.nbr_channels):
                    pcoef[filter_index,chanel, :] = Pstruct[chanel]['U3'][filter_index, :] * Pstruct[chanel]['lmbda'][:]; 

        start = time.time()
        for image_index in range(batch_size):
                # Convolve image with index image_index in the batch
                self.convolve_one_image(input_images[image_index,:,:,:],
                              one_image_shape,
                              Pstruct, pcoef,
                              filter_shape, 
                              image_index)   
        end = time.time()
       
        self.t_conv = (end - start)*1000/ batch_size 
        print 'convolution time of batch image {0}'.format(self.t_conv)
   #     print 'before downsample', self.input4D
        # downsample each feature map individually, using maxpooling
        start = time.time()
        pooled_out = downsample.max_pool_2d(input=self.input4D,
                                            ds=poolsize, 
                                            ignore_border=True)
                
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        sb = theano.shared(b)        
        self.output = T.tanh(pooled_out + sb.dimshuffle('x', 0, 'x', 'x'))
        end = time.time()
        self.t_downsample_activ = (end - start)*1000/ image_shape[0] 
        print 'pool+tanh time of batch image {0}'.format(self.t_downsample_activ)        
        return self.output.eval()
    
    """TODO change to have an image such as nbr channels as well"""
    def convolve_one_image(self, one_image, img_shape, 
                           Pstruct, pcoef, filter_shape,
                           image_index):
   
        rank = Pstruct[0]['U1'].shape[1]
        fwidth = Pstruct[0]['U1'].shape[0]
        fheight = Pstruct[0]['U2'].shape[0]
        #
     
        num_input_feature_maps = img_shape[0]
        U1 = np.ndarray((num_input_feature_maps, rank ,1, fwidth))
        U2 = np.ndarray((num_input_feature_maps, rank, fheight, 1))
        for chanel in range(num_input_feature_maps):
           U1[chanel,:,0, :] =  np.transpose(Pstruct[chanel]['U1']);  
           U2[chanel,:,:, 0] = np.transpose(Pstruct[chanel]['U2']);
        n_rows = img_shape[1] - fwidth + 1
        n_cols = img_shape[2] - fheight + 1
       # horizontal_conv_out = np.ndarray((img_shape[1], n_cols))
        vertical_conv_out = np.ndarray(( n_rows, n_cols,num_input_feature_maps, rank))
        start = time.time()
      #  vertical_filter_shape = (rank, fheight,1)
      #  vertical_filters = np.ndarray(vertical_filter_shape)      
      #  horizontal_filter_shape = (rank, 1, fwidth)
      #  horizontal_filters = np.ndarray(horizontal_filter_shape)
        for chanel in xrange(num_input_feature_maps):        
                for r in xrange(rank):
                   #   horizontal_conv_out = scipy.signal.convolve2d(one_image[chanel,:,:], 
                   #                                           U1[chanel,r,:,:], mode='valid')
                      vertical_conv_out[:,:, chanel,r] = scipy.signal.convolve2d(scipy.signal.convolve2d(one_image[chanel,:,:], 
                                                              U1[chanel,r,:,:], mode='valid'), 
                                                    U2[chanel, r,:,:], mode='valid')
        end = (time.time() - start)*1000
        print 'part 1 ', end
       
       
        start = time.time()       
        output_image = np.zeros((num_input_feature_maps,n_rows, n_cols))
        for filter_index in xrange(self.nbr_filters):            
         #   for chanel in xrange(num_input_feature_maps):
         #       temp = vertical_conv_out[chanel,:,:,:]*pcoef[filter_index, chanel,:]
         #       output_image[chanel, :, :] = np.sum(temp, axis=2)
            output_image = vertical_conv_out[:,:,:,:]*pcoef[filter_index,:,:]                
            self.input4D[image_index,filter_index,:,:] =  np.sum(output_image, axis=(3, 2))
        
        end = (time.time() - start)*1000;
        print 'part2 ', end
