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

from load_data import load_mnist_data
from google.protobuf import text_format


from lenet_conv_pool_layer import LeNetConvPoolLayer
from lenet_layer_conv_pool_non_symbolic import LeNetLayerConvPoolNonSymbolic
from lenet_layer_conv_pool_separable_non_symbolic import LeNetLayerConvPoolSeparableNonSymbolic
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

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
           self.create_layers_from_settings(settings);
           self.initialize_logger()
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
        self.batch_size = 100
        
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
    
    def initialize_logger(self):
        (file_path, extension) = os.path.splitext(self.cached_weights_file)
        logger_filename = file_path
        d = datetime.datetime.now()
        # add calendar day information
        logger_filename += '_' + str(d.day) + '_' + str(d.month) + '_' + str(d.year);
        # add hour information
        logger_filename += '_' + str(d.hour) + '_' + str(d.minute) + '_' + str(d.second);
        # add extension
        logger_filename += '.log'        
        logging.basicConfig(filename=logger_filename, level=logging.DEBUG)
            
        
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
        self.n_test_batches = 5
        timings = []
        test_losses = np.zeros((self.n_test_batches, 1))
        for batch_index in xrange(self.n_test_batches):
               print 'batch nr', batch_index
               # Create tine object
               cnn_time = CNNTime()   
               start = time.time()
               test_losses[batch_index] = self.process_batch(batch_index, cnn_time)
               endt = (time.time() - start)*1000/(self.batch_size)
               cnn_time.t_total = round(endt, 2)
               print cnn_time.to_string()
               logging.debug(cnn_time.to_string())
               timings.append(cnn_time)
               
        self.log_cnn_time_summary(timings) 
        
        test_score = np.mean(test_losses)
        test_score *= 100        
        print ' Test error of best ', test_score
        logging.debug('Test error: ' + str(test_score))
        
        
    def log_cnn_time_summary(self, timings):
        t_convolution = []
        t_downsample_activation = []
        t_non_conv_layers = 0
        t_total = 0
        N = len(timings[0].t_convolution)
        t_convolution = np.zeros((N, 1))
        t_downsample_activation = np.zeros((N, 1))
        for cnn_time in timings:
            t_total += cnn_time.t_total
            t_non_conv_layers += cnn_time.t_non_conv_layers
            for i in range(N):
                t_convolution[i] += cnn_time.t_convolution[i]            
                t_downsample_activation[i] += cnn_time.t_downsample_activation[i]
                
                
        for i in range(N):
                t_convolution[i] /= self.n_test_batches         
                t_downsample_activation[i] /= self.n_test_batches       
        t_total /= self.n_test_batches
        t_non_conv_layers /= self.n_test_batches
        logging.debug('Final average results')
        for i in range(N):         
               logging.debug(' convolution '+ str(i) + ' : ' + str(t_convolution[i])) 
               logging.debug(' downs + activ '+ str(i) + ' : ' + str(t_downsample_activation[i])) 
        logging.debug(' non conv layers : ' + str(t_non_conv_layers))         
        logging.debug(' total avg image time : ' + str(t_total)) 
        
                
    def process_batch(self, batch_index, cnn_time):
        """ Process one single batch"""
        self.x = self.test_set_x[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        self.y = self.test_set_y[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
 
        layer_input = self.x.reshape((self.batch_size, 1, self.input_shape[0], self.input_shape[1]))
        pooled_W = self.input_shape[0];
        pooled_H = self.input_shape[1];
        nbr_feature_maps = 1
        iter = 0
        for clayer_params in self.convolutional_layers:
           if clayer_params.HasField('rank') == False:
           #     print 'Iter ', iter
                layer_output = self.layer_convolutional.run_batch(layer_input, 
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                        W = self.cached_weights[iter + 1], 
                                        b = self.cached_weights[iter],
                                        poolsize=(self.poolsize, self.poolsize)).eval()
#                layer_output = LeNetConvPoolLayer(self.rng, input = layer_input, 
#                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
#                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
#                                                     clayer_params.filter_w, clayer_params.filter_w),
#                                       poolsize=(self.poolsize, self.poolsize),
#                                        W = self.cached_weights[iter + 1], 
#                                        b = theano.shared(self.cached_weights[iter]))
              #  print 'LAYER OUTPUT IS'
              #  print layer_output    
                cnn_time.t_convolution.append(round(self.layer_convolutional.convolutional_time,2))
                cnn_time.t_downsample_activation.append(round(self.layer_convolutional.downsample_time,2))
                                        
           else:
         #       print 'Separable ', iter
                layer_output = self.layer_separable_convolutional.run_batch(
                                                    layer_input, 
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       Pstruct = self.cached_weights[iter + 1], 
                                       b = self.cached_weights[iter],
                                       poolsize=(self.poolsize, self.poolsize))
                cnn_time.t_convolution.append(round(self.layer_separable_convolutional.t_conv,2))
                cnn_time.t_downsample_activation.append(round(self.layer_separable_convolutional.t_downsample_activ,2))
        #   print 'image_shape ', self.batch_size, nbr_feature_maps, pooled_W, pooled_H
        #   print 'filter_shape ', clayer_params.num_filters, nbr_feature_maps, clayer_params.filter_w, clayer_params.filter_w
           pooled_W = (pooled_W - clayer_params.filter_w + 1) / self.poolsize;
           pooled_H = (pooled_H - clayer_params.filter_w + 1) / self.poolsize;
           layer_input = layer_output;
           nbr_feature_maps = clayer_params.num_filters;
           iter += 2

     
#        # construct a fully-connected sigmoidal layer
        start = time.time()
        nbr_input = nbr_feature_maps * pooled_W * pooled_H ## Why is this SO??        
        layer_input = layer_input.reshape((self.batch_size, nbr_input))
        hlayers = []
        for hlayer_params in self.hidden_layers:
            layer = HiddenLayer(self.rng, input=layer_input, n_in=nbr_input,
                         n_out = hlayer_params.num_outputs, activation= T.tanh,
                         W = self.cached_weights[iter +1], b = self.cached_weights[iter])
            nbr_input = hlayer_params.num_outputs;
            layer_input = layer.output
            hlayers.append(layer)
            iter += 2
#         
#        # classify the values of the fully-connected sigmoidal layer
        output_layer = LogisticRegression(input=layer_input, n_in= nbr_input, n_out=10, 
                                               W = self.cached_weights[iter +1], 
                                               b = self.cached_weights[iter])
        cnn_time.t_non_conv_layers = (time.time() - start)*1000 / self.batch_size
        cnn_time.t_non_conv_layers = round(cnn_time.t_non_conv_layers, 2)
        print 'last layers image time final {0} in ms '.format(cnn_time.t_non_conv_layers)
        
        result =  output_layer.errors(self.y)
        print 'result is ', result.eval()
        return result.eval()

## Store average timing per batch
class CNNTime:
    def __init__(self):
        ## Convolution time per layer
        self.t_convolution = []
        ## Downsampling time per layer
        self.t_downsample_activation = []
        self.t_non_conv_layers = 0
        self.t_total = 0
    def to_string(self):
        return "Time :%s :%s %5.2f %5.2f" % (str(self.t_convolution), 
                                         str(self.t_downsample_activation),
                                         self.t_non_conv_layers, 
                                         self.t_total)
    