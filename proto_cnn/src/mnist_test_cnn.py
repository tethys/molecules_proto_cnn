# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:39:22 2014

@author: vivianapetrescu
"""

import convolutional_neural_network_settings_pb2 as pb_cnn
import numpy as np
import logging
import datetime
import os
import time
import theano.tensor as T
import theano

from google.protobuf import text_format

from test_tp_cnn import CNNTestTP
from lenet_conv_pool_layer import LeNetConvPoolLayer
from lenet_layer_conv_pool_non_symbolic import LeNetLayerConvPoolNonSymbolic
from lenet_layer_conv_pool_separable_non_symbolic import LeNetLayerConvPoolSeparableNonSymbolic
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

class CNNTestTPmnist(CNNTestTP):
    def __init__(self, protofile, cached_weights_file, frame = None):
		self.frame = frame
		super(CNNTestTPmnist, self).__init__(protofile, cached_weights_file)

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
    	if self.small_data == False:
		# Upscale the data
    		N = 10000
    		tmp_images = np.zeros((N, 56,56))
    		for i in range(N):
        		tmp_images[i,:,:] = scipy.misc.imresize(test_set[0][i,:,:], (56,56)) 
    		test_set_2 = (tmp_images, test_set[1])
    		test_set_x, test_set_y = shared_dataset(test_set_2)
    
	else:
		test_set_x, test_set_y = shared_dataset(test_set)
    	rval = (test_set_x, test_set_y)
    	return rval
