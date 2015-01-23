#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:19:55 2014

@author: vivianapetrescu
"""

import argparse
import sys
from mitocondria_train_cnn import CNNTrainVOCMitocondria
from mitocondria_test_cnn import CNNTestVOCMitocondria
from cnn_separable_non_symbolic import ConvolutionalNeuralNetworkNonSymbolic;
from convolutional_nnet_train import ConvolutionalNeuralNetworkTrain;
from convolutional_nnet_test import ConvolutionalNeuralNetworkTest;
from convolutional_neural_network_separable import ConvolutionalNeuralNetworkSeparableTest;
def main():
    sys.path.append('/Users/vivianapetrescu/Documents/theano_tut/convolutional-neural-net/');
    print 'Number of arguments:', len(sys.argv)
    print 'Arguments list:', str(sys.argv)

     
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--proto_file', help="path to proto file", dest='prototxt_file', required=True)
    parser.add_argument('-w','--cached_weights_file', help="path to weights file", dest='cached_weights_file', required=True)
    parser.add_argument('-r','--run_mode', help="mode 0 for train\n 1 for test, 2 for train and test", dest='mode', type = int, required=True)
    results = parser.parse_args()
    
    print 'my file is', results.prototxt_file, 'ups'
    print 'weights file is', results.cached_weights_file
    print 'run mode', results.mode

    if results.mode == 0 or results.mode == 2: # train model
	small_model = False
    	cnn = CNNTrainVOCMitocondria(results.prototxt_file, results.cached_weights_file, small_model)
    	cnn.build_model()
    	cnn.train_model()
    if results.mode == 1 or results.mode == 2: # test model
    	cnn = ConvolutionalNeuralNetworkSeparableTest(results.prototxt_file, results.cached_weights_file)
    	cnn.build_model()
    	cnn.test_model()
    if results.mode == 3:
      cnn = CNNTestVOCMitocondria(results.prototxt_file, results.cached_weights_file)
      cnn.compute_test_error()

if __name__ == '__main__':
    main()




