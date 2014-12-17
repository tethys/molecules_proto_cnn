# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:20:00 2014

@author: vivianapetrescu
"""

import argparse
import logging
import numpy as np
import sys
import mlab
import pymatlab
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
from sktensor import dtensor, ktensor, cp_als
import matplotlib.pyplot as plt
import matplotlib 

def main():
    matplotlib.use('Qt4Agg', warn=False)
    sys.path.append('/Users/vivianapetrescu/Documents/theano_tut/convolutional-neural-net/');
    sys.path.append('/Users/vivianapetrescu/Downloads/matlab_wrapper')
    print 'Number of arguments:', len(sys.argv)
    print 'Arguments list:', str(sys.argv)
 
    prototxt_file = 'experiments/test_separable_decomposition/cnn_small_test.prototxt' 
    cached_weights_file = 'experiments/test_separable_decomposition/cnn_model_original2.npy' 
    print 'protofile is', prototxt_file 
    print 'weights file is', cached_weights_file

    run(prototxt_file, cached_weights_file)
### Load cached weights
### Load settings
### Run for every convolutional layer separable filters and save it somewhere.
### Save also the other parameters with it

def run(prototxt_file, cached_weights_file):
        """Load settings"""        
        settings = pb_cnn.CNNSettings();        
        try:        
           f = open(prototxt_file, "r")
           data=f.read()
           print 'Protofile content:'
           print data
           text_format.Merge(data, settings);
           f.close();
        except IOError:
           print "Could not open file " + prototxt_file;
        """ Load weights"""
        cached_weights = np.load(cached_weights_file)
        params = decompose_layers(settings, cached_weights)
      #  np.save(options.sep_weights_file, params)
        
def decompose_layers(settings, cached_weights):
    """ Run through every conv layer and check were we have rank enabled.
        If we have it enabled, run decompose tensor.
    """
     # TODO this
    params = [];
    N = cached_weights.size - 1
    for layer in settings.conv_layer:
        params.append(cached_weights[N])
        P_struct = decompose_tensor(cached_weights[N - 1])
        params.append(P_struct)
        N = N - 2
        break
        
    for i in range(N, -1, -1):
             params.append(cached_weights[i])
    params.reverse() 
    print 'cached weights'
    for w in cached_weights:
        print w.size
    return params

def decompose_tensor(filters):
    """ filters is of type input feature maps, output feature maps, wxh of filter
        Output is a structure P which contains lambda, U{1}, U{2}, U{3}    
    """
    # Set logging to DEBUG to see CP-ALS information
    logging.basicConfig(level=logging.DEBUG)
    print filters.shape
    filters = np.array(filters)   
    print filters.shape 
    print filters.dtype
    nbr_filters = filters.shape[0]
    fwidth = filters.shape[2]
    fheight = filters.shape[3]
    Pstruct = []
    for chanel in range(filters.shape[1]):
        filter_for_channel = filters[:,chanel,:,:]
        filter_for_channel.reshape(nbr_filters, fwidth, fheight)
        filter_for_channel = np.swapaxes(filter_for_channel, 0,2);
        print 'Number of filters ', nbr_filters
        print 'filter_for channel shape ', filter_for_channel.shape
        fig, axes = plt.subplots(nrows=5, ncols=4)
        fig.tight_layout() 
        
        for f in xrange(nbr_filters):
            img = filter_for_channel[:,:,f]
            plt.subplot(5,4,f)
            plt.imshow(img)
        plt.show(block=False)
        T  = dtensor(filter_for_channel);
        rank = np.floor(nbr_filters*0.6);
        print 'rank is ', rank
        session = pymatlab.session_factory()
        session.putvalue('A',rank)
        del session
        ## P.U, P.lmbda
        print 'P U0,U1,U2, lambda sizes: ', P.U[0].size, P.U[1].size, P.U[2].size, P.lmbda
        print 'fit was ', fit        
        Pstruct.append(P)
        #dtensor(ktensor(U).toarray())
        print np.allclose(T, P.totensor())
    
    
    U = [np.random.rand(i,3) for i in (20, 10, 14)]
    
    Tn = dtensor(ktensor(U).toarray())
    P, fit, itr, _ = cp_als(Tn, 10)
    print 'P U0,U1,U2, lambda sizes: ', P.U[0].size, P.U[1].size, P.U[2].size, P.lmbda
    print 'fit was ', fit  
    print np.allclose(Tn, P.totensor())
    
    return Pstruct
if __name__ == '__main__':
    main()