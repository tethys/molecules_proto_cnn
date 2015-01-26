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

from base_cnn import CNNBase
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
from lenet_conv_pool_layer import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


class CNNRetrain(CNNBase):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, protofile, cached_weights):
	self.cnntype = 'RETRAIN'
	super(CNNRetrain, self).__init__(protofile, cached_weights)
	self.load_weights()
    
    def build_model(self):
	# Fixed rng, make the results repeatable

        datasets = self.load_samples()

	# Train, Validation, Test 100000, 20000, 26... fot Mitocondria set
        # Train, Validation, Test 50000, 10000, 10000 times 28x28 = 784 for MNIST dataset
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        # assumes the width equals the height
        img_width_size = numpy.sqrt(self.test_set_x.shape[1].eval()).astype(int)
        print "Image shape %s x %s" % (img_width_size, img_width_size)
        self.input_shape = (img_width_size, img_width_size)

        # Compute number of minibatches for training, validation and testing
        # Divide the total number of elements in the set by the batch size
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= self.batch_size
        self.n_valid_batches /= self.batch_size
        self.n_test_batches /= self.batch_size

        #print('Size train %d, valid %d, test %d' % (self.train_set_x.shape.eval(), self.valid_set_x.shape.eval(), self.test_set_x.shape.eval())
        print('Size train_batches %d, n_valid_batches %d, n_test_batches %d' % (self.n_train_batches, self.n_valid_batches, self.n_test_batches))

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
              		         # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print 'Building the model ...'
	for i in xrange(len(self.cached_weights)):
		self.cached_weights[i] = theano.shared(self.cached_weights[i])
        # The input is an 4D array of size, number of images in the batch size, number of channels
	# (or number of feature maps), image width and height.
	nbr_feature_maps = 1
	layer_input = self.x.reshape((self.batch_size, nbr_feature_maps, self.input_shape[0], self.input_shape[1]))
        pooled_W = self.input_shape[0];
        pooled_H = self.input_shape[1];
	# Add convolutional layers followed by pooling
        clayers = []
	iter = 0
        for clayer_params in self.convolutional_layers:
            print 'Adding conv layer nbr filter %d, Ksize %d' % (clayer_params.num_filters, clayer_params.filter_w)
            layer = LeNetConvPoolLayer(self.rng, input = layer_input,
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps,
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize),
                                       W = self.cached_weights[iter + 1], b = self.cached_weights[iter])
	    iter = iter + 2
            clayers.append(layer)
            pooled_W = (pooled_W - clayer_params.filter_w + 1) / self.poolsize;
            pooled_H = (pooled_H - clayer_params.filter_w + 1) / self.poolsize;
            layer_input = layer.output;
            nbr_feature_maps = clayer_params.num_filters;


        # Flatten the output of the previous layers and add
        # fully connected sigmoidal layers	
        layer_input = layer_input.flatten(2);
        nbr_input = nbr_feature_maps * pooled_W * pooled_H
        hlayers = []
        for hlayer_params in self.hidden_layers:
            print 'Adding hidden layer fully connected %d' % (hlayer_params.num_outputs)
            layer = HiddenLayer(self.rng, input=layer_input, n_in=nbr_input,
                         n_out = hlayer_params.num_outputs, activation=T.tanh,
                         W = self.cached_weights[iter +1], b = self.cached_weights[iter])
            nbr_input = hlayer_params.num_outputs;
            layer_input = layer.output
            hlayers.append(layer)
            iter+=2

        # classify the values of the fully-connected sigmoidal layer
        self.output_layer = LogisticRegression(input=layer_input, n_in= nbr_input, 
						n_out = self.last_layer.num_outputs, W = self.cached_weights[iter+1], 
						b = self.cached_weights[iter]) 

        # the cost we minimize during training is the NLL of the model
        self.cost = self.output_layer.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.output_layer.params
        for hl in reversed(hlayers):
            self.params += hl.params
        for cl in reversed(clayers):
            self.params += cl.b_params

        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

    def retrain_model(self):
	raise NotImplementedError()
    def save_parameters(self):
          weights = [i.get_value(borrow=True) for i in self.best_params]
	  ## add here the interleaved convolutional layers
          nbr_hidden_layers = size(hlayers)
	  toskip = 1 + nbr_hidden_layers * 2
	  retrainedw = []
	  for i in xrange(toskip):
		retrainedw.append(weights[i])          
          for c in xrange(size(clayers)):
                retrainedw.append(self.cached_weights[2*c + 1])
		retrainedw.append(weights[toskip])
		toskip++
          numpy.save(self.cached_weights_file +'retrain.npy', weights)
