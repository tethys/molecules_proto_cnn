# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetresc 
"""

import datetime
import logging
import numpy as np
import os
import scipy
import time

import theano
import theano.tensor as T

from retrain_voc_cnn import CNNRetrainVOC
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
from lenet_conv_pool_layer import LeNetConvPoolLayer
from load_mitocondria import load_mitocondria
from load_data_rescaled import load_mnist_data_rescaled
from load_data import load_mnist_data
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


class CNNRetrainVOCMitocondria(CNNRetrainVOC):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, cnn_settings_protofile, cached_weights_file, small_set):
	self.small_set = small_set	
	super(CNNRetrainVOCMitocondria, self).__init__(cnn_settings_protofile, cached_weights_file)

    def load_samples(self):
	print 'Value of small set is ', self.small_set
	
        # Load datasets
        path_to_data = '../data/'
        if self.small_set == True:
     	 	train_set_x = np.load(path_to_data + 'train_set_x_100000_51.npy')
     	 	train_set_y = np.load(path_to_data + 'train_set_y_100000_51.npy')
     	 	valid_set_x = np.load(path_to_data + 'valid_set_x_20000_51.npy')
     	 	valid_set_y = np.load(path_to_data +'valid_set_y_20000_51.npy')
        else:
     	 	train_set_x = np.load(path_to_data + 'train_set_x_1000000_51.npy')
	#	train_set_x = train_set_x[0:300000,:]
     	 	train_set_y = np.load(path_to_data + 'train_set_y_1000000_51.npy')
     	 	valid_set_x = np.load(path_to_data + 'valid_set_x_200000_51.npy')
	#	train_set_y = train_set_y[0:300000]
     	 	valid_set_y = np.load(path_to_data + 'valid_set_y_200000_51.npy')
     
        test_set_x = np.load(path_to_data +'test_set_x_fr1_51.npy')
        test_set_y = np.load(path_to_data +'test_set_y_fr1_51.npy')

        train_set_x, train_set_y = self.shared_dataset((train_set_x, train_set_y))
        valid_set_x, valid_set_y = self.shared_dataset((valid_set_x, valid_set_y))
        test_set_x, test_set_y = self.shared_dataset((test_set_x, test_set_y))
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval
