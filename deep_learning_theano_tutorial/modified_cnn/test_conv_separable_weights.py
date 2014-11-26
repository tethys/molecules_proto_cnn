# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 15:29:48 2014

@author: vivianapetrescu
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:50:01 2014

@author: vivianapetrescu
"""
import numpy as np
import scipy
import theano
from theano.tensor.nnet import conv
from theano import tensor as T
import matplotlib.pyplot as plt

def main():

    ## Load image
    img_shape = (1,20, 28,28)
    np.random.seed(100)
    image = np.random.rand(1,20,28,28)* 10;
    sys.path.append('')
    ## Load filter 1
    original_weights = np.load('experiments/test_one_layer/cnn_model_original2.npy')
    ## Load separable filter 2
    separable_weights = np.load('experiments/test_separable_decomposition/separable_filters_model_original.npy');

    ## use convolution mode 1
    origw = original_weights[4][0,:,:,:];
    origw = origw.reshape(1,20,5,5)
    print origw.shape
    conv_out = conv.conv2d(input = image, filters = origw, 
                filter_shape = origw.shape, image_shape = img_shape)
    conv_out.reshape((24,24))
    print conv_out.eval()
    print conv_out.shape.eval()
    print type(conv_out)
   # plt.set_cmap('gray')
   # plt.imshow(conv_out)        
   # plt.show()
   # plt.savefig('original_filters.png')    
    
#use convolution mode 2

    Pstruct = separable_weights[4]
    print Pstruct[0]['U1'].shape
    print Pstruct[0]['U2'].shape
    print Pstruct[0]['U3'].shape
    rank = Pstruct[0]['U1'].shape[1]
    fwidth = Pstruct[0]['U1'].shape[0]
    fheight = Pstruct[0]['U2'].shape[0]
    nbr_filters = Pstruct[0]['U3'].shape[0]
#
#    #   rank 4, w,h 3x3, nbr filter 7
    print 'num input feaure maps ', origw.shape[1]
    num_input_feature_maps = 20
    
    n_rows = img_shape[2]- fwidth + 1
    n_cols = img_shape[3] - fheight + 1
    output_image = np.zeros((num_input_feature_maps,n_rows, n_cols))
    horizontal_conv_out = np.zeros((rank, num_input_feature_maps, img_shape[2],n_cols))
    vertical_conv_out = np.zeros((rank, num_input_feature_maps, n_rows, n_cols))
    for chanel in range(num_input_feature_maps):        
            horizontal_filter_shape = (rank, 1,fwidth)
            horizontal_filters = np.ndarray(horizontal_filter_shape)
            horizontal_filters[:, 0, :] = np.transpose(Pstruct[chanel]['U1']);
            print horizontal_conv_out.shape
            ## rank x im size, im fsize
            for r in range(rank):
              horizontal_conv_out[r,chanel,:,:] = scipy.signal.convolve2d(image[0,chanel,:,:], 
                                                          horizontal_filters[r,:,:], mode='valid')
#    
            vertical_filter_shape = (rank, fheight,1)
            vertical_filters = np.ndarray(vertical_filter_shape)        
            vertical_filters[:,:, 0] = np.transpose(Pstruct[chanel]['U2']);
            for r in range(rank):            
              vertical_conv_out[r,chanel,:,:] = scipy.signal.convolve2d(horizontal_conv_out[r,chanel,:,:], 
                                                vertical_filters[r,:,:], mode='valid')
##        ## numberof images, number of filters, image width, image height

            temp = np.zeros((n_rows, n_cols))
            alphas = Pstruct[chanel]['U3']
            f = 0
            for r in range(rank):
                     out = vertical_conv_out[r,chanel, :,:]* alphas[f, r] * Pstruct[chanel]['lmbda'][r]; 
                     temp = temp + out
            output_image[chanel, :, :] = temp
    output_image = np.sum(output_image, axis=0)
    print output_image

if __name__ == '__main__':
    main()