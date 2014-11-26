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
    def run_batch(self, input_images, image_shape, filter_shape,  W, b, poolsize):
        assert image_shape[1] == filter_shape[1]
        # the bias is a 1D tensor -- one bias per output feature map
        # convolve input feature maps with filters
        Pstruct = W
        batch_size = image_shape[0]             
        fwidth = Pstruct[0]['U1'].shape[0]
        fheight = Pstruct[0]['U2'].shape[0]
        nbr_channels = image_shape[1]
        nbr_filters = Pstruct[0]['U3'].shape[0]
        initial_n_rows = image_shape[2]
        initial_n_cols = image_shape[3]
        
        # Final number of rows and columns        
        final_n_rows = initial_n_rows - fwidth + 1
        final_n_cols = initial_n_cols - fheight + 1
        # The convolved input images
        self.input4D = np.zeros((batch_size, nbr_filters, final_n_rows, final_n_cols))
        print 'batch size ', batch_size        
        one_image_shape = (initial_n_rows, initial_n_cols)
       # assert one_image_shape == (1,28,28)
        for image_index in range(batch_size):
            for channel_index in range(nbr_channels):
                # Convolve image with index image_index in the batch
                print 'convolution start'
                self.input4D = self.convolve_one_image(input_images[image_index,channel_index,:,:],
                              one_image_shape,
                              Pstruct, 
                              filter_shape, 
                              image_index,
                              channel_index)   
                print 'convolution end'
    
   #     print 'before downsample', self.input4D
        # downsample each feature map individually, using maxpooling
        start = time.time()
        pooled_out = downsample.max_pool_2d(input=self.input4D,
                                            ds=poolsize, 
                                            ignore_border=True)
        end = time.time()
        self.downsample_time = (end - start)*1000/ image_shape[0]
        print 'after downsampling'
                
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        sb = theano.shared(b)        
        self.output = T.tanh(pooled_out + sb.dimshuffle('x', 0, 'x', 'x'))
        return self.output.eval()
    
    """TODO change to have an image such as nbr channels as well"""
    def convolve_one_image(self, one_image, image_shape, 
                           Pstruct, filter_shape,
                           image_index,
                           channel_index):
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
         
         
         # Construct horizontal filters
         #TODO save the filters in the correct shape
         horizontal_filter_shape = (rank, 1, fwidth)
         horizontal_filters = np.ndarray(horizontal_filter_shape)
         horizontal_filters[:, 0, :] = np.transpose(Pstruct[channel_index]['U1']);
         initial_n_rows = image_shape[0]
         final_n_rows = initial_n_rows- fwidth + 1
         final_n_cols = image_shape[1] - fheight + 1 
         
         # Output is 1 x rank x W x H
         start = time.time()
         horizontal_conv_out = np.zeros((rank, initial_n_rows, final_n_cols))
         for r in range(rank):
             horizontal_conv_out[r,:,:] = scipy.signal.convolve2d(one_image, 
                                               horizontal_filters[r,:,:], mode='valid')
                                           
         end = time.time()
         downsample_time = (end - start)*1000
         print 'horizontal conv ', downsample_time
     #    print 'horinzontal ', horizontal_conv_out.eval()
     #    print 'input ', one_image
         
         # Construct vertical filters
         vertical_filter_shape = (rank, fheight, 1)
         vertical_filters = np.ndarray(vertical_filter_shape)        
         vertical_filters[:,:, 0] = np.transpose(Pstruct[channel_index]['U2']);


         conv_out = np.zeros((rank, final_n_rows, final_n_cols))
         start = time.time()
         for r in range(rank):
             # temp is 1x1x imageW x imageH
#             conv_out[r,:,:] = conv.conv2d(input = horizontal_conv_out[:,r,:,:], 
#                             filters = vertical_filters[r,:,:],
#                             filter_shape = (1, fheight, 1), 
#                             image_shape = (1, initial_n_rows, final_n_cols)).eval()[0,:,:]
            # conv_out[r,:,:] = A[0,:,:]
            conv_out[r,:,:] = scipy.signal.convolve2d(horizontal_conv_out[r,:,:], 
                                                     vertical_filters[r,:,:], mode='valid')
              
         end = time.time()
         downsample_time = (end - start)*1000
         print 'vertical conv ', downsample_time
   #      print 'before combinig'
   #      print conv_out
         
         nbr_filters = Pstruct[0]['U3'].shape[0]
         # Final number of rows and columns                        
         ## numberof images, number of filters, image width, image height
         alphas = Pstruct[channel_index]['U3']  
         start = time.time()
         for f in range(nbr_filters):            
            temp = np.zeros((final_n_rows, final_n_cols))
            for r in range(rank):
                temp = temp + conv_out[r, :,:]* alphas[f, r] * Pstruct[channel_index]['lmbda'][r]; 
            #input4D =T.set_subtensor(input4D[image_index,f,:,:], temp)
            self.input4D[image_index,f,:,:] = temp  
         end = time.time()
         
         downsample_time = (end - start)*1000
         print 'final time ', downsample_time
         return self.input4D