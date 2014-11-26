# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:39:22 2014

@author: vivianapetrescu
"""
import time

import theano
import convolutional_neural_network_settings_pb2 as pb_cnn
import numpy as np

from load_data import load_mnist_data
from google.protobuf import text_format

from lenet_layer_conv_pool_non_symbolic import LeNetLayerConvPoolNonSymbolic
from lenet_layer_conv_pool_separable_non_symbolic import LeNetLayerConvPoolSeparableNonSymbolic
import theano.tensor as T
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from lenet_conv_pool_layer import LeNetConvPoolLayer

class ConvolutionalNeuralNetworkNonSymbolic:
    def __init__(self, cnn_settings_protofile, cached_weights_file):
        """ Update parameters from protobuffer file"""
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
        self.load_weights()
           
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
        self.output_layer = settings.last_layer;

        # required parameter
        self.cost_function = settings.cost_function;
        self.input_shape = (28,28); # this is the size of MNIST images
    

        ## Create Layers    
        self.rng = np.random.RandomState(23455)
        self.layer_convolutional = LeNetLayerConvPoolNonSymbolic(self.rng)
        self.layer_separable_convolutional = LeNetLayerConvPoolSeparableNonSymbolic(self.rng)
   #     self.hidden_layer = HiddenLayer(rng)
    
    def load_weights(self):
        # Load weights...
        weights = np.load(self.cached_weights_file)
        #   sp.io.savemat('temporary.mat', {'w',weights})
        self.cached_weights = []        
        for w in reversed(weights):
            self.cached_weights.append(w)
            print 'weight array size ', len(w)
        print 'cached weights size is ', len(self.cached_weights)
        
        
    def compute_test_error(self):
        """Loop through the batches and run process_batch"""
        
        # Load the data
        datasets = load_mnist_data(self.dataset)

        # Train, Validation, Test 50000, 10000, 10000 times 28x28 = 784
        test_set_x, test_set_y = datasets[2]
        # shape of test_set_x is 10000x784
        #shape of test_set_y is 10000x784
        self.test_set_x = test_set_x.get_value()
        self.test_set_y = test_set_y

        # compute number of minibatches for training, validation and testing
        # 50000/50 = 1000, 10000/50 = 200
        self.n_test_batches = self.test_set_x.shape[0]
        self.n_test_batches /= self.batch_size
        print 'nbr batches, batch size ', self.n_test_batches, self.batch_size
        print self.test_set_y

        
        print 'Running test'
        self.n_test_batches = 1
        test_losses = np.zeros((self.n_test_batches, 1))
        for batch_index in xrange(self.n_test_batches):
               start = time.time()
               print 'batch nr', batch_index
               test_losses[batch_index] = self.process_batch(batch_index + 1)
               endt = (time.time() - start)*1000/self.batch_size
               print 'image time {0} in ms '.format(endt)
               
        test_score = np.mean(test_losses)
        print ' test error of best ', test_score * 100.

    def process_batch(self, batch_index):
        """ Process one single batch"""
        
        self.x = self.test_set_x[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        self.y = self.test_set_y[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
 
        layer_input = self.x.reshape((self.batch_size, 1, self.input_shape[0], self.input_shape[1]))
        pooled_W = self.input_shape[0];
        pooled_H = self.input_shape[1];
        nbr_feature_maps = 1
        iter = 0
        for clayer_params in self.convolutional_layers:
           print clayer_params.num_filters
           print clayer_params.filter_w
           print 'Inside loop '           
           if clayer_params.HasField('rank') == False:
                print 'Iter ', iter
#                layer_output = self.layer_convolutional.run_batch(layer_input, 
#                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
#                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
#                                                     clayer_params.filter_w, clayer_params.filter_w),
#                                        W = self.cached_weights[iter + 1], 
#                                        b = self.cached_weights[iter],
#                                        poolsize=(self.poolsize, self.poolsize)).eval()
                layer_output = LeNetConvPoolLayer(self.rng, input = layer_input, 
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize),
                                        W = self.cached_weights[iter + 1], 
                                        b = theano.shared(self.cached_weights[iter]))
              #  print 'LAYER OUTPUT IS'
              #  print layer_output    
                                        
           else:
                print 'Separable ', iter
                layer_output = self.layer_separable_convolutional.run_batch(
                                                    layer_input, 
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       W = self.cached_weights[iter + 1], 
                                       b = self.cached_weights[iter],
                                       poolsize=(self.poolsize, self.poolsize))
              #  print 'LATER finished ', layer_output
           print 'image_shape ', self.batch_size, nbr_feature_maps, pooled_W, pooled_H
           print 'filter_shape ', clayer_params.num_filters, nbr_feature_maps, clayer_params.filter_w, clayer_params.filter_w
           pooled_W = (pooled_W - clayer_params.filter_w + 1) / self.poolsize;
           pooled_H = (pooled_H - clayer_params.filter_w + 1) / self.poolsize;
           layer_input = layer_output;
           nbr_feature_maps = clayer_params.num_filters;
           iter += 2
           

#        
#        
#        # construct a fully-connected sigmoidal layer
        layer_input = layer_input.flatten(2);
        nbr_input = nbr_feature_maps * pooled_W * pooled_H ## Why is this SO??
        hlayers = []
        for hlayer_params in self.hidden_layers:
            print hlayer_params.num_outputs
            layer = HiddenLayer(self.rng, input=layer_input, n_in=nbr_input,
                         n_out = hlayer_params.num_outputs, activation= T.tanh,
                         W = self.cached_weights[iter +1], b = self.cached_weights[iter])
            nbr_input = hlayer_params.num_outputs;
            layer_input = layer.output
            hlayers.append(layer)
            iter += 2
#         
        print 'nbr inputs ', nbr_input 
#        # classify the values of the fully-connected sigmoidal layer
        output_layer = LogisticRegression(input=layer_input, n_in= nbr_input, n_out=10, 
                                               W = self.cached_weights[iter +1], 
                                               b = self.cached_weights[iter])


        result =  output_layer.errors(self.y)
        print 'result is ', result.eval()
        return result.eval()
