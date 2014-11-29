# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:17:50 2014

@author: vivianapetrescu
"""
import numpy as np
import theano
import time

import scipy
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class LeNetLayerConvPoolNonSymbolic:
    def __init__(self, rng):
        self.rng = rng
    
    def run_batch(self, images, filter_shape, image_shape, W=None, b=None, poolsize=(2, 2)):
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
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(
                self.rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                               borrow=True)
        else:
            self.W = W

        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        return self.run_conv_pool(images, filter_shape, image_shape, poolsize)
        
        
    def run_conv_pool(self, images, filter_shape, image_shape, poolsize):    
        # convolve input feature maps with filters
        # W is 50, 20, 5, 5 (nbr_filters, nbr_channels, fwidht, fheight)
       # print 'filter shape  ', filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]
       # print 'image shape ', image_shape[0], image_shape[1], image_shape[2], image_shape[3]  

        batch_size = image_shape[0]             
        fwidth = self.W.shape[2]
        fheight = self.W.shape[3]
        nbr_channels = self.W.shape[1]
        nbr_filters = self.W.shape[0]
        initial_n_rows = image_shape[2]
        initial_n_cols = image_shape[3]
       
#        # Final number of rows and columns        
        final_n_rows = initial_n_rows - fwidth + 1
        final_n_cols = initial_n_cols - fheight + 1
 #       conv_out = conv.conv2d(input=images, filters=self.W,
 #               filter_shape=filter_shape, image_shape=image_shape)
        start = time.time()
        conv_out = np.zeros((batch_size, nbr_filters, final_n_rows, final_n_cols))
        for b in range(batch_size):         
            for f in range(nbr_filters):
                temp = np.zeros((final_n_rows, final_n_cols))
                for c in range(nbr_channels):
                    temp += scipy.signal.convolve2d(images[b,c,:,:], self.W[f,c,:,:], mode='valid')
                conv_out[b,f,:,:] = temp
                
        end = time.time()
        self.convolutional_time = (end - start)*1000/image_shape[0]                
        start = time.time()
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        end = time.time()
        self.downsample_time = (end - start)*1000/ batch_size
        
        print 'conv {0}, {1} ms'.format(self.convolutional_time, self.downsample_time)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        sb = theano.shared(self.b)
        return T.tanh(pooled_out + sb.dimshuffle('x', 0, 'x', 'x'))

   