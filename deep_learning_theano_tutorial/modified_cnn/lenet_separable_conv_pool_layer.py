# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 00:19:05 2014

@author: vivianapetrescu
"""

import scipy.io
import numpy as np
import time

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.signal import conv

class LeNetSeparableConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input_images, filter_shape, image_shape, poolsize=(2, 2),  
                 Pstruct = None, b= None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature/channels maps,
                             image height, image width)
                             For a gray image, this is 1 for the first layer.
                             For the following layers it is equal with the 
                             number of output  channels/feature maps

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        # the bias is a 1D tensor -- one bias per output feature map
        # convolve input feature maps with filters


        batch_size = image_shape[0]             
        fwidth = Pstruct[0]['U1'].shape[0]
        fheight = Pstruct[0]['U2'].shape[0]
        nbr_filters = Pstruct[0]['U3'].shape[0]
        # Final number of rows and columns        
        n_rows = image_shape[2]- fwidth + 1
        n_cols = image_shape[3] - fheight + 1
        input4D = theano.shared(np.zeros((batch_size, nbr_filters, 
                                          n_rows, n_cols)))
        print 'batch size ', batch_size        
        one_image_shape = (image_shape[1], image_shape[2], image_shape[3])
        assert one_image_shape == (1,28,28)
        for image_index in range(batch_size):
            input4D = self.convolve_one_image(input4D, input_images[image_index,:,:,:], 
                              one_image_shape,
                              Pstruct, 
                              filter_shape, 
                              image_index)   
        # downsample each feature map individually, using maxpooling
        start = time.time()
        pooled_out = downsample.max_pool_2d(input=input4D,
                                            ds=poolsize, 
                                            ignore_border=True)
        end = time.time()
        self.downsample_time = (end - start)*1000/ image_shape[0]
        
                
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
    
    """TODO change to have an image such as nbr channels as well"""
    def convolve_one_image(self,input4D, one_image, image_shape, Pstruct, filter_shape,
                           image_index):
         """
        Convolve one image with separabl filters.

        :type input: theano.tensor.dtensor3
        :param input: symbolic image tensor, of shape image_shape

        :type image_shape: tuple or list of length 3
        :param image_shape: ( nbr channels, image height, image width)
        
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters,nbrmaps, filter height,filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """                        
  
    
        ## We look at the composition for the first channel in the beginning  
         rank = Pstruct[0]['U1'].shape[1]
         fwidth = filter_shape[2]
         fheight = filter_shape[3]
        
         
         horizontal_filter_shape = (rank, 1, fwidth)
         horizontal_filters = np.ndarray(horizontal_filter_shape)
         num_input_feature_maps = 1
         for chanel in range(num_input_feature_maps):
            horizontal_filters[:, 0, :] = np.transpose(Pstruct[chanel]['U1']);
        
        # output is 1 x rank x W x H
         horizontal_conv_out = conv.conv2d(input=one_image, 
                                             filters = horizontal_filters,
                                             filter_shape = horizontal_filter_shape, 
                                             image_shape = image_shape)
        
         
         vertical_filter_shape = (rank, fheight, 1)
         vertical_filters = np.ndarray(vertical_filter_shape)        
         for chanel in range(num_input_feature_maps):
            vertical_filters[:,:, 0] = np.transpose(Pstruct[chanel]['U2']);
    #    number of filters, num input feature maps,
     ## TODO fix assignment here!!     
         new_image_shape = (rank, image_shape[1], image_shape[2] -fwidth + 1)
         n_rows = image_shape[1]- fwidth + 1
         n_cols = image_shape[2] - fheight + 1 
         conv_out = theano.shared(np.zeros((rank, n_rows, n_cols)))
         for r in range(rank):
             # temp is 1x1x imageW x imageH
             A = conv.conv2d(input = horizontal_conv_out[:,r,:,:], 
                                filters = vertical_filters[r,:,:],
                                filter_shape = (1, fheight, 1), 
                                image_shape = (1, image_shape[1], image_shape[2] -fwidth + 1))
             conv_out = T.set_subtensor(conv_out[r,:,:], A[0,:,:])
  
         nbr_filters = Pstruct[0]['U3'].shape[0]
         # Final number of rows and columns        
                  
         ## numberof images, number of filters, image width, image height
         for f in range(nbr_filters):            
            temp = theano.shared(np.zeros((n_rows, n_cols)))
            for chanel in range(num_input_feature_maps):
                 alphas = Pstruct[chanel]['U3']
                 for r in range(rank):
                     out = conv_out[r, :,:]* alphas[f, r] * Pstruct[chanel]['lmbda'][r];   
                     temp = temp + out
            input4D =T.set_subtensor(input4D[image_index,f,:,:], temp)
         return input4D   


