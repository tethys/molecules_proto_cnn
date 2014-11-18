# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:06:13 2014

@author: vivianapetrescu
"""

import test_separable_filters


import argparse
import logging
import numpy as np
import sys
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
import matplotlib.pyplot as plt
from scipy import linalg

def main():
    sys.path.append('/Users/vivianapetrescu/Documents/theano_tut/convolutional-neural-net/');    
    
#    U = [np.random.rand(i,3) for i in (20, 10, 14)]
#    
#    np.random.seed(1014)
#    U = np.random.randn(20,5,5);
#    fig, axes = plt.subplots(nrows=5, ncols=4)
#    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
#    for i in xrange(20):
#         img = U[i,:,:]
#   #     print img.shape
#         plt.subplot(5,4,i)
#         plt.imshow(img)
#         
#    plt.show()
#    plt.savefig('temp.png')
#    
#Compare the two decompositions    
    original_weights = np.load('experiments/test_separable_decomposition/cnn_model_original2.npy'); 
    print original_weights.size
    separable_weights = np.load('experiments/test_separable_decomposition/separable_filters_model_original.npy');
    print separable_weights.size
    for i in range(original_weights.size):
       print 'weights shape ', original_weights[i].shape  
    for i in range(separable_weights.size):
       print 'weights len ', len(separable_weights[i])    
    ## Plot now weights 7 and 5
   
    ## 4 and 6 are the new values.
    sep_filters6 = separable_weights[6];
    ## returns smth times rank
    print sep_filters6[0]['U1'].shape
    print sep_filters6[0]['U2'].shape
    print sep_filters6[0]['U3'].shape        
    
    fig, axes = plt.subplots(nrows=5, ncols=4)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    origw = original_weights[6]; 
    origw = origw.reshape((20,5,5))
    plt.set_cmap('gray')
    for i in xrange(20):
         img = origw[i,:,:]
         img = img.reshape((5,5))
#   #     print img.shape
         plt.subplot(5,4,i)
         plt.imshow(img)
#         
    plt.show()
    plt.savefig('original_filters.png')
 
    ## 4 and 6 are the new values.
    sep_filters4 = separable_weights[4];
    ## returns smth times rank
    print sep_filters4[0]['U1'].shape
    print sep_filters4[0]['U2'].shape
    print sep_filters4[0]['U3'].shape      
     
    filters_no = 20
    rank = 18
    coefs = np.zeros((rank, filters_no));
    filters_size = 5
    recomp = np.zeros((filters_size, 
                       filters_size, 
                       filters_no));
    P = sep_filters6[0];
    
    sep = np.zeros((filters_size,filters_size,rank));
    normSep = np.zeros(rank);
    for j in range(rank):
        print 'P is ', P['U1'][:,j].shape
        sep[:,:,j] =  np.outer(P['U1'][:,j], P['U2'][:,j]);
        temp = sep[:,:,j];
        if j == 0:
            print 'temporary ', temp
        temp.reshape((filters_size, filters_size));
        normSep[j] = linalg.norm(temp, 2)
        print 'norm should be 1 ', normSep[j]
        sep[:,:,j] = sep[:,:,j]/normSep[j];
        
    # Recompose the tensor
    for i in xrange(filters_no):
        sum_recomp = 0;
        for j in range(rank):
            sum_recomp = sum_recomp + P['lmbda'][j]*normSep[j]*sep[:,:,j]*P['U3'][i,j];
            coefs[j,i] = P['lmbda'][j] * P['U3'][i,j]* normSep[j];
        recomp[:,:,i] = sum_recomp;
         
    fig, axes = plt.subplots(nrows=5, ncols=4)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    for i in xrange(20):
         img = recomp[:,:,i]
         img = img.reshape((5,5))
#   #     print img.shape
         plt.subplot(5,4,i)
         plt.imshow(img)
    plt.show()
    plt.savefig('separable_filters.png')
if __name__ == '__main__':
    main()