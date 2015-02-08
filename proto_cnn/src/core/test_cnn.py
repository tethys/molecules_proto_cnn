# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:39:22 2014

@author: vpetresc
"""

import numpy as np
import logging
import time
import theano.tensor as T

from base_cnn import CNNBase
from lenet_conv_pool_layer import LeNetConvPoolLayer
from lenet_layer_conv_pool_non_symbolic import LeNetLayerConvPoolNonSymbolic
from lenet_layer_conv_pool_separable_non_symbolic import LeNetLayerConvPoolSeparableNonSymbolic
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

class CNNTest(CNNBase):
    """The class is responsible for creating the layers of the network.
        The loading of the data and the type of the error used is defined in
        derived classes.

    Args:

    Returns:

    """
    def __init__(self, cnn_settings_protofile, cached_weights_file):
        """ Update parameters from protobuffer file"""
        self.cnntype = 'TEST'
        super(CNNTest, self).__init__(cnn_settings_protofile, cached_weights_file)
        self.load_weights()
        ## Create Layers
        self.test_set_x = None
        self.test_set_y = None
        self.x = None
        self.y = None
        self.input_shape = (0, 0)
        self.n_test_batches = 0

    def test_model(self):
        """Loop through the batches and run process_batch"""

        # Load the data
        datasets = self.load_samples()

        # Train, Validation, Test 50000, 10000, 10000 times 28x28 = 784
        test_set_x, test_set_y = datasets
        self.test_set_x = test_set_x.get_value()
        self.test_set_y = test_set_y

        #"Image width is the square root of the x dimenstion of the test x data"
        image_width_size = np.sqrt(test_set_x.shape[1].eval()).astype(int)
        self.input_shape = (image_width_size, image_width_size)

        # compute number of minibatches for training, validation and testing
        # 50000/50 = 1000, 10000/50 = 200
        self.n_test_batches = self.test_set_x.shape[0]
        self.n_test_batches /= self.batch_size
        logging.debug('nbr batches %d, batch size %d' % (self.n_test_batches, self.batch_size))
        print self.test_set_y

        print 'Running test'
        timings = []
        resultlist = [dict() for x in xrange(self.n_test_batches)]
        for batch_index in xrange(self.n_test_batches):
            print 'batch nr', batch_index
            # Create tine object
            cnn_time = CNNTime()
            start = time.time()
            resultlist[batch_index] = self.process_batch(batch_index, cnn_time)
            endt = (time.time() - start)*1000/(self.batch_size)
            cnn_time.t_total = round(endt, 2)
            print 'Batch time ', cnn_time.to_string()
            logging.debug(cnn_time.to_string())
            timings.append(cnn_time)
            err = self.compute_batch_error(resultlist[batch_index])
            print 'Batch error ', err

        self.log_cnn_time_summary(timings)
        self.log_fit_info()
        test_score = self.compute_all_saples_error(resultlist)
        print ' Test error of best ', test_score*100
        logging.debug('Test error: ' + str(test_score))
        return test_score

    def process_batch(self, batch_index, cnn_time):
        """Process one single batch

        Args:
          batch_index: 
          cnn_time: 

        Returns:

        """
        start = time.time()
        self.x = self.test_set_x[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        self.y = self.test_set_y[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        nbr_feature_maps = 1
        layer_input = self.x.reshape((self.batch_size,
                                      nbr_feature_maps,
                                      self.input_shape[0], self.input_shape[1]))
        pooled_width = self.input_shape[0]
        pooled_height = self.input_shape[1]
        idx_weight = 0

        start = time.time()
        for clayer_params in self.convolutional_layers:
            if clayer_params.HasField('rank') == False:
                startl = time.time()
                layer_convolutional = LeNetLayerConvPoolNonSymbolic(self.rng)
                #self.layer_separable_convolutional =
                #LeNetLayerConvPoolSeparableNonSymbolic(self.rng)
                layer_output = layer_convolutional.run_batch(
                               layer_input,
                                image_shape=(self.batch_size, nbr_feature_maps, pooled_width, pooled_height),
                                 filter_shape=(clayer_params.num_filters, nbr_feature_maps,
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                 W = self.cached_weights[idx_weight + 1],
                                 b = self.cached_weights[idx_weight],
                                 poolsize=(self.poolsize, self.poolsize)).eval()
#                layer_output = LeNetConvPoolLayer(self.rng, input = layer_input,
#                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_width, pooled_height),
#                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps,
#                                                     clayer_params.filter_w, clayer_params.filter_w),
#                                       poolsize=(self.poolsize, self.poolsize),
#                                        W = self.cached_weights[iter + 1],
#                                        b = theano.shared(self.cached_weights[iter]))
              #  print 'LAYER OUTPUT IS'
              #  print layer_output
                cnn_time.t_convolution.append(round(layer_convolutional.convolutional_time, 2))
                cnn_time.t_downsample_activation.append(round(layer_convolutional.downsample_time, 2))
                endt = (time.time() - startl)*1000 / self.batch_size
                print 'layer time %.2f' % endt
            else:
         #       print 'Separable ', iter
                layer_separable_convolutional = LeNetLayerConvPoolSeparableNonSymbolic(self.rng)
                layer_output = layer_separable_convolutional.run_batch(
                                                    layer_input,
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_width, pooled_height),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps,
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       Pstruct = self.cached_weights[idx_weight + 1],
                                       b = self.cached_weights[idx_weight],
                                       poolsize=(self.poolsize, self.poolsize)).eval()
                cnn_time.t_convolution.append(round(layer_separable_convolutional.t_conv, 2))
                cnn_time.t_downsample_activation.append(round(layer_separable_convolutional.t_downsample_activ, 2))
        #   print 'image_shape ', self.batch_size, nbr_feature_maps, pooled_width, pooled_height
        #   print 'filter_shape ', clayer_params.num_filters, nbr_feature_maps, clayer_params.filter_w, clayer_params.filter_w
            pooled_width = (pooled_width - clayer_params.filter_w + 1) / self.poolsize
            pooled_height = (pooled_height - clayer_params.filter_w + 1) / self.poolsize
            layer_input = layer_output
            nbr_feature_maps = clayer_params.num_filters
            idx_weight += 2
        endt = (time.time() - start)*1000 / self.batch_size
        print 'total layers time %.2f in ms ' % endt

        # construct a fully-connected sigmoidal layer
        start = time.time()
        nbr_input = nbr_feature_maps * pooled_width * pooled_height ## Why is this SO??
        layer_input = layer_input.reshape((self.batch_size, nbr_input))
        hlayers = []
        for hlayer_params in self.hidden_layers:
            layer = HiddenLayer(self.rng, input=layer_input, n_in=nbr_input,
                         n_out=hlayer_params.num_outputs, activation=T.tanh,
                         W=self.cached_weights[idx_weight +1], b=self.cached_weights[idx_weight])
            nbr_input = hlayer_params.num_outputs
            layer_input = layer.output
            hlayers.append(layer)
            idx_weight += 2

        # classify the values of the fully-connected sigmoidal layer
        self.output_layer = LogisticRegression(input=layer_input, n_in=nbr_input,
                                               n_out=self.last_layer.num_outputs,
                                               W=self.cached_weights[idx_weight + 1],
                                               b=self.cached_weights[idx_weight])
        cnn_time.t_non_conv_layers = (time.time() - start)*1000 / self.batch_size
        cnn_time.t_non_conv_layers = round(cnn_time.t_non_conv_layers, 2)

        return self.output_layer.result_count_dictionary(self.y)

    def log_fit_info(self):
        """Logs the fit of the separable filters for the separable layers.
            The information is stored in the numpy arrays

        Args:

        Returns:

        """
        idx = 0
        for clayer_params in self.convolutional_layers:
            if clayer_params.HasField('rank') == False:
                logging.debug('-')
            else:
                ## log mean and std of fit per chanel
                sep_filters_struct = self.cached_weights[idx + 1]
                N = len(sep_filters_struct)
                fitarray = np.zeros((N, 1))
                for chanel in xrange(N):
                    print sep_filters_struct[chanel]['fit']
                    fitarray[chanel] = sep_filters_struct[chanel]['fit']
                logging.debug('Fit mean {0}, std {1}'.format(np.mean(fitarray), np.std(fitarray)))
            idx += 2

    def log_cnn_time_summary(self, timings):
        """The method takes as input a vector of timings
            and aggregates the results

        Args:
          timings: 

        Returns:

        """
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

    def compute_batch_error(self, batch_result_dict):
        """Abstract method for computing the error per batch (VOC or error rate)

        Args:
          batch_result_dict: 

        Returns:

        """
        raise NotImplementedError()
    def compute_all_samples_error(self, all_samples_result):
        """Abstract method for computing the error on all test samples

        Args:
          all_samples_result: 

        Returns:

        """
        raise NotImplementedError()


## Store average timing per batch
class CNNTime(object):
    """The class holds the timings for every layer,
        (e.g. convolutional, hidden) etc

    Args:

    Returns:

    """
    def __init__(self):
        """ For every type of layer there is an array
            where the timings are added """
        ## Convolution time per layer
        self.t_convolution = []
        ## Downsampling time per layer
        self.t_downsample_activation = []
        self.t_non_conv_layers = 0
        self.t_total = 0
    def to_string(self):
        """Printable version the the class contents"""
        return "Time :%s :%s %5.2f %5.2f" % (str(self.t_convolution),
                                             str(self.t_downsample_activation),
                                             self.t_non_conv_layers,
                                             self.t_total)
