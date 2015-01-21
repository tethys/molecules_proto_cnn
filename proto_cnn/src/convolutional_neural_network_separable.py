# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 00:30:04 2014

@author: vivianapetrescu
"""


import os
import sys
import time

import scipy as sp
import numpy

import theano
import theano.tensor as T

from load_mitocondria import load_mitocondria
from load_data import load_mnist_data
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from lenet_conv_pool_layer import LeNetConvPoolLayer
from lenet_separable_conv_pool_layer import LeNetSeparableConvPoolLayer
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format

class ConvolutionalNeuralNetworkSeparableTest(object):
    def __init__(self, cnn_settings_protofile, cached_weights_file):
        settings = pb_cnn.CNNSettings();        
        self.cached_weights_file = cached_weights_file
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
        self.batch_size = 1
        
       # self.nbr_convolutional_layers = settings.conv_layer.size();
       # self.nbr_hidden_layers = settings.hidden_layer.size();
      
        for layer in settings.conv_layer:
              self.convolutional_layers.append(layer)
        for layer in settings.hidden_layer:
              self.hidden_layers.append(layer)      

        # required at least one layer
        self.last_layer = settings.last_layer;

        # required parameter
        self.cost_function = settings.cost_function;

    def build_model(self):
        rng = numpy.random.RandomState(23455)

        datasets = load_mitocondria()

        # Train, Validation, Test 50000, 10000, 10000 times 28x28 = 784
        self.test_set_x, self.test_set_y = datasets[2]

        img_width_size = numpy.sqrt(self.test_set_x.shape[1].eval()).astype(int)
        print "Image shape ", img_width_size
        self.input_shape = (img_width_size, img_width_size); # this is the size of MNIST images
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
        weights = numpy.load(self.cached_weights_file)
     #   sp.io.savemat('temporary.mat', {'w',weights})
        cached_weights = []        
        for w in reversed(weights):
            cached_weights.append(w)
            print 'weight array size ', len(w)
        print 'cached weights size is ', len(cached_weights)
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
            print 'Inside loop '
            if clayer_params.HasField('rank') == False:
                print 'Iter ', iter
                print cached_weights[iter+1]
                layer = LeNetConvPoolLayer(rng, input = layer_input, 
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize),
                                        W = cached_weights[iter + 1], 
                                        b = theano.shared(cached_weights[iter]))
            else:
                print 'Separable ', iter
                layer = LeNetSeparableConvPoolLayer(rng, input_images = layer_input, 
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize),
                                       Pstruct = cached_weights[iter + 1], 
                                       b = theano.shared(cached_weights[iter]))
            print 'image_shape ', self.batch_size, nbr_feature_maps, pooled_W, pooled_H
            print 'filter_shape ', clayer_params.num_filters, nbr_feature_maps, clayer_params.filter_w, clayer_params.filter_w
            clayers.append(layer)
            pooled_W = (pooled_W - clayer_params.filter_w + 1) / self.poolsize;
            pooled_H = (pooled_H - clayer_params.filter_w + 1) / self.poolsize;
            layer_input = layer.output;
            nbr_feature_maps = clayer_params.num_filters;
            iter += 2
        
        
        # construct a fully-connected sigmoidal layer
        layer_input = layer_input.flatten(2);
        nbr_input = nbr_feature_maps * pooled_W * pooled_H ## Why is this SO??
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
         
        print 'nbr inputs ', nbr_input 
        # classify the values of the fully-connected sigmoidal layer
        self.output_layer = LogisticRegression(input=layer_input, n_in= nbr_input, n_out= self.last_layer.num_outputs, 
                                               W = cached_weights[iter +1], 
                                               b = cached_weights[iter])


    def test_model(self):
          print 'Running test'
             # create a function to compute the mistakes that are made by the model
          test_model_result = theano.function([self.index], self.output_layer.VOC_values(self.y),
             givens={
                self.x: self.test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]},
                name='cnn_test_model' , on_unused_input='ignore')
            ###############
            # TEST MODEL #
            ###############
          print '... testing'
 
          # test it on the test set
          
          test_losses = numpy.zeros((100, 1))
          for i in xrange(100):
               start = time.time()
               print 'batch nr', i
               test_losses[i] = test_model_result(i)
               endt = (time.time() - start)*1000/self.batch_size
               print 'image time {0} in ms '.format(endt)
               
          test_score = sp.stats.nanmean(test_losses)
          print test_losses
	  print ' test error of best non nan ', test_score * 100.

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]
