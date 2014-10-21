# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014

@author: vivianapetrescu
"""
import sys
import os
import convolutional_neural_network_settings_pb2 as pb_cnn
# Parameters: make this a protocol buffer
#  learning_rate=0.1, 
#  n_epochs=2,
#  dataset='mnist.pkl.gz',
#  batch_size=50
#  poolsize 2 x 2
#  Layer1 conv
#     20, 5x5
#  Layer2 conv
#     50, 5x5
#  Layer3 full
#     500 tanh
#  Layer4 full - last
#     10 
# Cost negative log likelihood

# Make the network read this and setup
class ConvolutionalNeuralNetwork(object):
    def __init__(self, cnn_settings_protofile):
        # Itearte through every layer
        settings = pb_cnn.CNNSettings();
#        try:        
#            f = open(cnn_settings_protofile, "rb")
#            settings.ParseFromString(f.read());
#            f.close();
#        except IOError:
#            print "Could not open file " + cnn_settings_protofile;
        return None
    def train():
        return 0;
    def save_parameters():
        return 0;
    
        
    
    
    
    