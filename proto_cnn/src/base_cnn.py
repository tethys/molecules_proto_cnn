# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetresc

TODO consider using abstract base class
http://zaiste.net/2013/01/abstract_classes_in_python/
"""

import datetime
import logging
import numpy
import os
import scipy
import time

import theano
import theano.tensor as T

import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
from lenet_conv_pool_layer import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


class CNNBase(object):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, cnn_settings_protofile, cached_weights_file):
        # Filename of the .npy file where the trained weights will be saved.
	self.cached_weights_file = cached_weights_file
        self.initialize_logger()
        # Create protobuff empty object
        settings = pb_cnn.CNNSettings();
        try:
           f = open(cnn_settings_protofile, "r")
           data = f.read()
           text_format.Merge(data, settings)
           print "Network settings are \n"
           print data
           print "\n"
	   logging.info(data)
           # Build up the ConvolutionalNeuralNetworkTrain model from the layers description
           self.create_layers_from_settings(settings)
	   f.close();
        except IOError:
           print "Could not open file " + cnn_settings_protofile;

    def create_layers_from_settings(self, settings):
        # Default values for optional parameters
        self.learning_rate = 0.1;
        self.n_epochs = 100;
	self.batch_size = 100;
	self.poolsize = 2;
        # This fields are optional
        ##TODO add colored warning if default value is used
        if settings.HasField('learning_rate'):
                self.learning_rate = settings.learning_rate;
        else:
		print 'Warning - default learning rate ', self.learning_rate
	if settings.HasField('n_epochs'):
                self.n_epochs = settings.n_epochs
	else:
		print 'Warning - default number of epochs ', self.n_epochs
        if settings.HasField('batch_size'):
                self.batch_size = settings.batch_size
	else:
		print 'Warning - default number of batch size ', self.batch_size
        if settings.HasField('poolsize'):
                self.poolsize = settings.poolsize;
	else:
		print 'Warning - default number of poolsize ', self.poolsize
        # This fields are required
        self.dataset = 'mnist.pkl.gz' #settings.dataset;

        # Add every convolutional layer to an array. Similarly for hidden layers.
        self.convolutional_layers = [];
        self.hidden_layers = [];
        for layer in settings.conv_layer:
              self.convolutional_layers.append(layer)
        for layer in settings.hidden_layer:
              self.hidden_layers.append(layer)

        # required at least one layer
        self.output_layer = settings.last_layer;

	# Required parameters TODO still needed?
        self.dataset = settings.dataset
        self.cost_function = settings.cost_function;


    def initialize_logger(self):
	# The log file is saved in the same folder of the cached weights file
        (file_path, extension) = os.path.splitext(self.cached_weights_file)
        d = datetime.datetime.now()
        # add calendar day information
        # add hour information
        # add extension
	logger_filename = "%s_%d_%d_%d_%d_%d_%d.log" % (file_path, d.day, d.month, d.year,
							d.hour, d.minute, d.second)
        logging.basicConfig(filename=logger_filename, level=logging.DEBUG)


    def load_samples(self):
	raise NotImplementedError()
