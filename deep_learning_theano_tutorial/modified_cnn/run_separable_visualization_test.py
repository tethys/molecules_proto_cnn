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
from sktensor import dtensor, ktensor, cp_als
import matplotlib.pyplot as plt

def main():
    sys.path.append('/Users/vivianapetrescu/Documents/theano_tut/convolutional-neural-net/');    
    
    U = [np.random.rand(i,3) for i in (20, 10, 14)]
    
    np.random.seed(1014)
    U = np.random.randn(20,5,5);
    fig, axes = plt.subplots(nrows=5, ncols=4)
    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    for i in xrange(20):
         img = U[i,:,:]
   #     print img.shape
         plt.subplot(5,4,i)
         plt.imshow(img)
         
    plt.show()
    plt.savefig('temp.png')
    Tn = dtensor(U)
   # Tn = dtensor(ktensor(U).toarray())
    P, fit, itr, _ = cp_als(Tn, 4)
    print 'P U0,U1,U2, lambda sizes: ', P.U[0].size, P.U[1].size, P.U[2].size, P.lmbda
    print 'fit was ', fit  
    
    print np.allclose(Tn, P.totensor())
    
    
    

    
if __name__ == '__main__':
    main()