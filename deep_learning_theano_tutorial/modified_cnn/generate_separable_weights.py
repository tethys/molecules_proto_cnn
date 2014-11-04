# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:59:25 2014

@author: vivianapetrescu
"""

import argparse
import logging
import numpy as np
import sys
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
from sktensor import dtensor, cp_als

def main():
    sys.path.append('/Users/vivianapetrescu/Documents/theano_tut/convolutional-neural-net/');
    print 'Number of arguments:', len(sys.argv)
    print 'Arguments list:', str(sys.argv)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--proto_file', help="path to proto file", dest='prototxt_file', required=True)
    parser.add_argument('-w','--cached_weights_file', help="path to weights file", dest='cached_weights_file', required=True)
    parser.add_argument('-s','--separable_weights_file', help="path to separable weights file", dest='sep_weights_file',  required=True)
    options = parser.parse_args()
    
    print 'protofile is', options.prototxt_file 
    print 'weights file is', options.cached_weights_file
    print 'separable weights file', options.sep_weights_file

    run(options)
### Load cached weights
### Load settings
### Run for every convolutional layer separable filters and save it somewhere.
### Save also the other parameters with it

def run(options):
        """Load settings"""        
        settings = pb_cnn.CNNSettings();        
        try:        
           f = open(options.prototxt_file, "r")
           data=f.read()
           print 'Protofile content:'
           print data
           text_format.Merge(data, settings);
           f.close();
        except IOError:
           print "Could not open file " + options.prototxt_file;
        """ Load weights"""
        cached_weights = np.load(options.cached_weights_file)
        params = decompose_layers(settings, cached_weights)
        np.save(options.sep_weights_file, params)
        
def decompose_layers(settings, cached_weights):
    """ Run through every conv layer and check were we have rank enabled.
        If we have it enabled, run decompose tensor.
    """
     # TODO this
    params = [];
    N = cached_weights.size - 1
    for layer in settings.conv_layer:
        params.append(cached_weights[N])
        if layer.HasField('rank'):
             P_struct = decompose_tensor(cached_weights[N - 1], layer.rank)
             params.append(P_struct)
        else:
             params.append(cached_weights[N - 1])
        N = N - 2
    for i in range(N, -1, -1):
             params.append(cached_weights[i])
    params.reverse() 
    for p in params:    
        print p
    print 'cached weights'
    for w in cached_weights:
        print w.size
    return params

def decompose_tensor(filters, rank):
    """ filters is of type input feature maps, output feature maps, wxh of filter
        Output is a structure P which contains lambda, U{1}, U{2}, U{3}    
    """
    # Set logging to DEBUG to see CP-ALS information
    logging.basicConfig(level=logging.INFO)
    print filters.shape
    filters = np.array(filters)   
    print filters.shape 
    print filters.dtype
    a = np.arange(6)
    print a.dtype
    nbr_filters = filters.shape[0]
    fwidth = filters.shape[2]
    fheight = filters.shape[3]
    Pstruct = []
    for chanel in range(filters.shape[1]):
        filter_for_channel = filters[:,chanel,:,:]
        filter_for_channel.reshape(nbr_filters, fwidth, fheight)
        P, fit, itr, exectimes = cp_als(dtensor(filter_for_channel), rank, init='random')
        ## P.U, P.lmbda
        print 'P U0,U1,U2, lambda sizes: ', P.U[0].size, P.U[1].size, P.U[2].size, P.lmbda
        Pstruct.append(P)
    return Pstruct
if __name__ == '__main__':
    main()
