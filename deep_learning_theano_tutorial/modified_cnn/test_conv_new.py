# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 20:50:01 2014

@author: vivianapetrescu
"""
import numpy as np
import sys
import theano
from theano.tensor.signal import conv
from theano import tensor as T
import matplotlib.pyplot as plt

def main():

    ## Load image
    img_shape = (1,28,28)
    np.random.seed(100)
    image = np.random.rand(1,28,28)* 10;
    sys.path.append('')
    ## Load filter 1
    original_weights = np.load('experiments/test_separable_decomposition/cnn_model_original2.npy')
    ## Load separable filter 2
    separable_weights = np.load('experiments/test_separable_decomposition/separable_filters_model_original.npy');

    ## use convolution mode 1
    origw = original_weights[6][0,0,:,:];
    origw = origw.reshape(1,5,5)
    print origw.shape
    conv_out = conv.conv2d(input = image, filters = origw, 
                filter_shape = origw.shape, image_shape = img_shape)
    conv_out.reshape((24,24))
    conv_out = conv_out
    print conv_out.shape.eval()
    print conv_out.eval()

    print type(conv_out)
   # plt.set_cmap('gray')
   # plt.imshow(conv_out)        
   # plt.show()
   # plt.savefig('original_filters.png')    
    
#use convolution mode 2

    Pstruct = separable_weights[6]
    print Pstruct[0]['U1'].shape
    print Pstruct[0]['U2'].shape
    print Pstruct[0]['U3'].shape
    rank = Pstruct[0]['U1'].shape[1]
    fwidth = Pstruct[0]['U1'].shape[0]
    fheight = Pstruct[0]['U2'].shape[0]
    nbr_filters = Pstruct[0]['U3'].shape[0]

    #   rank 4, w,h 3x3, nbr filter 7
    print 'num input feaure maps ', origw.shape[1]
    num_input_feature_maps = 1
    horizontal_filter_shape = (rank, 1,fwidth)
    horizontal_filters = np.ndarray(horizontal_filter_shape)
    for chanel in range(num_input_feature_maps):
            horizontal_filters[:, 0, :] = np.transpose(Pstruct[chanel]['U1']);

    ## 1 x rank x im size, im fsize
    horizontal_conv_out = conv.conv2d(input = image, filters = horizontal_filters,
                               filter_shape = horizontal_filter_shape, image_shape = img_shape)
                
    print 'shape is ', horizontal_conv_out.shape.eval()
    
    vertical_filter_shape = (rank, fheight,1)
    vertical_filters = np.ndarray(vertical_filter_shape)        
    for chanel in range(num_input_feature_maps):
           vertical_filters[:,:, 0] = np.transpose(Pstruct[chanel]['U2']);
#    #    number of filters, num input feature maps,
    new_image_shape = (rank, img_shape[1], img_shape[2] -fwidth + 1)
    # output is rank x rank imgw, imgh. The input is  rank, im size, im size
    conv_out = conv.conv2d(input = horizontal_conv_out[0,:,:,:], filters = vertical_filters,
                              filter_shape = vertical_filter_shape, image_shape = new_image_shape)

#        ## numberof images, number of filters, image width, image height
    batch_size = 1;
    n_rows = img_shape[1]- fwidth + 1
    n_cols = img_shape[2] - fheight + 1

    print conv_out.shape.eval()
    for f in range(1):            
          temp = np.zeros((n_rows, n_cols))
          for chanel in range(num_input_feature_maps):
                 alphas = Pstruct[chanel]['U3']
                 for r in range(rank):
                     out = conv_out[r,r, :,:]* alphas[f, r] * Pstruct[0]['lmbda'][r]; 
                     temp = temp + out
#            if f == 0:
#                print'first map'
#                print temp
         # T.set_subtensor(input4D[:,f,:,:], temp)
         # print 'input shape ', input4D[:,f,:,:].shape
         # print 'temp shape ', temp.shape.eval()
         # for i in range(n_rows):
         #     for j in range(n_cols):
         #         input4D[0,0,i,j] = temp[0,i,j]
          print temp.eval()

if __name__ == '__main__':
    main()