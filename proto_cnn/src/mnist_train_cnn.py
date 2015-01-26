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

from train_tp_cnn import CNNTrainTP
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
from lenet_conv_pool_layer import LeNetConvPoolLayer
from load_mitocondria import load_mitocondria
from load_data_rescaled import load_mnist_data_rescaled
from load_data import load_mnist_data
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


class CNNTrainTPmnist(CNNTrainTP):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, cnn_settings_protofile, cached_weights_file, small_set):
	self.small_set = small_set	
	super(CNNTrainTPmnist, self).__init__(cnn_settings_protofile, cached_weights_file)

    def load_samples(self):
	print 'Value of small set is ', self.small_set
	
        # Load datasets
        dataset = 'mnist.pkl.gz'
       # Download the MNIST dataset if it is not present
    	data_dir, data_file = os.path.split(dataset)
    	if data_dir == "" and not os.path.isfile(dataset):
        	# Check if dataset is in the data directory.
       		 new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
           	 dataset = new_path

    	if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        	import urllib
        	origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        	print 'Downloading data from %s' % origin
        	urllib.urlretrieve(origin, dataset)

    	print '... loading data'

    	# Load the dataset
    	f = gzip.open(dataset, 'rb')
    	train_set, valid_set, test_set = cPickle.load(f)
    	f.close()
    	#train_set, valid_set, test_set format: tuple(input, target)
    	#input is an numpy.ndarray of 2 dimensions (a matrix)
    	#witch row's correspond to an example. target is a
    	test_set_x, test_set_y = shared_dataset(test_set)
    	valid_set_x, valid_set_y = shared_dataset(valid_set)
    	train_set_x, train_set_y = shared_dataset(train_set)

    	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    	return rval
