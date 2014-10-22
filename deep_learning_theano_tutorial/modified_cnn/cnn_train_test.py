#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:19:55 2014

@author: vivianapetrescu
"""
import sys
from convolutional_nnet_train import ConvolutionalNeuralNetworkTrain;
from convolutional_nnet_test import ConvolutionalNeuralNetworkTest;

def main():
    sys.path.append('/Users/vivianapetrescu/Documents/theano_tut/convolutional-neural-net/');
    cnn = ConvolutionalNeuralNetworkTrain('cnn_small_test.prototxt');
    cnn.build_model();
    cnn.train_model();
    cnn_test = ConvolutionalNeuralNetworkTest('cnn_small_test.prototxt')
    cnn_test.build_model()
    cnn_test.test_model()

if __name__ == '__main__':
    main()




