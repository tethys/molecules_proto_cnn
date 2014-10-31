# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:59:25 2014

@author: vivianapetrescu
"""

import argparse
import numpy as np
import sys
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format

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
        decompose_layers(settings, cached_weights)        
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
              params.append([])
        else:
             params.append(cached_weights[N - 1])
        N = N - 2
    for i in range(N, -1, -1):
             params.append(cached_weights[i])
    params = reversed(params)  
    for p in params:    
        print p.size
    print 'cached weights'
    for w in cached_weights:
        print w.size

def decompose_tensor():
    pass
if __name__ == '__main__':
    main()
