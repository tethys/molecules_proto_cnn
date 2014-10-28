# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014

@author: vivianapetrescu
"""

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from load_data import load_mnist_data
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from lenet_conv_pool_layer import LeNetConvPoolLayer

import sys
import os
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format
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
class ConvolutionalNeuralNetworkTest(object):
    def __init__(self, cnn_settings_protofile):
        settings = pb_cnn.CNNSettings();
        try:        
           f = open(cnn_settings_protofile, "r")
           # Itearte through every layer        
           data=f.read()
           print data
           text_format.Merge(data, settings);
           print "Network settings are "
           print  settings.__str__
           self.create_layers_from_settings(settings);
           f.close();
        except IOError:
           print "Could not open file " + cnn_settings_protofile;
        
    def create_layers_from_settings(self, settings):  
           # Default values for optionl parameters       
        self.learning_rate = 0.9;
        self.n_epochs = 5;
        # This fields are required
        self.dataset = 'mnist.pkl.gz' #settings.dataset;
        self.batch_size = 100;
        self.poolsize = 2;
        
        if settings.HasField('learning_rate'):
                self.learning_rate = settings.learning_rate;
        if settings.HasField('n_epochs'):
                self.n_epochs = settings.n_epochs;        
        if settings.HasField('batch_size'):
                self.batch_size = settings.batch_size;
        if settings.HasField('poolsize'):
                self.poolsize = settings.poolsize;
                
        # TODO this
        self.convolutional_layers = [];
        self.hidden_layers = [];
        
       # self.nbr_convolutional_layers = settings.conv_layer.size();
       # self.nbr_hidden_layers = settings.hidden_layer.size();
      
        for layer in settings.conv_layer:
              self.convolutional_layers.append(layer)
        for layer in settings.hidden_layer:
              self.hidden_layers.append(layer)      

        # required at least one layer
        self.output_layer = settings.last_layer;

        # required parameter
        self.cost_function = settings.cost_function;
        self.cached_weights_file = settings.cached_weights_file
        self.input_shape = (28,28); # this is the size of MNIST images


    def build_model(self):
        rng = numpy.random.RandomState(23455)

        datasets = load_mnist_data(self.dataset)

        # Train, Validation, Test 50000, 10000, 10000 times 28x28 = 784
        self.test_set_x, self.test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        # 50000/50 = 1000, 10000/50 = 200
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches /= self.batch_size

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

        # Load weights...
        weights = numpy.load(self.cached_weights_file + '.npy')
    
        W3 = theano.shared(weights[0])
        b3 = theano.shared(weights[1])
        W2 = theano.shared(weights[2])
        b2 = theano.shared(weights[3])
        W1 = theano.shared(weights[4])
        b1 = theano.shared(weights[5])
        W0 = theano.shared(weights[6])
        b0 = theano.shared(weights[7])

        cached_weights = []        
        for w in reversed(weights):
            cached_weights.append(theano.shared(w))
            
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print 'Building the model ...'
   
    
        layer_input = self.x.reshape((self.batch_size, 1, self.input_shape[0], self.input_shape[1]))
        pooled_W = self.input_shape[0];
        pooled_H = self.input_shape[1];
        nbr_feature_maps = 1
        clayers = []
        iter = 0
        for clayer_params in self.convolutional_layers:
            print clayer_params.num_filters
            print clayer_params.filter_w
            layer = LeNetConvPoolLayer(rng, input = layer_input, 
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize),
                                        W = cached_weights[iter + 1], b = cached_weights[iter])
            clayers.append(layer)
            pooled_W = (pooled_W - clayer_params.filter_w + 1) / self.poolsize;
            pooled_H = (pooled_H - clayer_params.filter_w + 1) / self.poolsize;
            layer_input = layer.output;
            nbr_feature_maps = clayer_params.num_filters;
            iter += 2
        
        
        # construct a fully-connected sigmoidal layer
        layer_input = layer_input.flatten(2);
        nbr_input = nbr_feature_maps * 4 * 4 ## Why is this SO??
        hlayers = []
        for hlayer_params in self.hidden_layers:
            print hlayer_params.num_outputs
            layer = HiddenLayer(rng, input=layer_input, n_in=nbr_input,
                         n_out = hlayer_params.num_outputs, activation=T.tanh,
                         W = cached_weights[iter +1], b = cached_weights[iter])
            nbr_input = hlayer_params.num_outputs;
            layer_input = layer.output
            hlayers.append(layer)
            iter += 2
            
        # classify the values of the fully-connected sigmoidal layer
        self.output_layer = LogisticRegression(input=layer_input, n_in= nbr_input, n_out=10)


    def test_model(self):
             # create a function to compute the mistakes that are made by the model
          self.test_model = theano.function([self.index], self.output_layer.errors(self.y),
             givens={
                self.x: self.test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]})
            ###############
            # TEST MODEL #
            ###############
          print '... testing'
 
          # test it on the test set
          test_losses = [self.test_model(i) for i in xrange(self.n_test_batches)]
          test_score = numpy.mean(test_losses)
          print(' test error of best ', test_score * 100.)





        
    
        
    
    
    
    
