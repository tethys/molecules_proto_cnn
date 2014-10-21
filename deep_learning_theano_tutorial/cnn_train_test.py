# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 16:19:55 2014

@author: vivianapetrescu
"""
from convolutional_nnet import ConvolutionalNeuralNetwork;

def main():
    sys.path.append('/Users/vivianapetrescu/Documents/theano_tut/convolutional-neural-net/');
    cnn = ConvolutionalNeuralNetwork('cnn_settings.proto');

if __name__ == '__main__':
    main()

