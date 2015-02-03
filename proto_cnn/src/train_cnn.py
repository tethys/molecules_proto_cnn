# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetresc

TODO consider using abstract base class
http://zaiste.net/2013/01/abstract_classes_in_python/
"""

import theano.tensor as T
import numpy

from base_cnn import CNNBase
from lenet_conv_pool_layer import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


class CNNTrain(CNNBase):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, protofile, cached_weights):
        self.cnntype = 'TRAIN'
        super(CNNTrain, self).__init__(protofile, cached_weights)
        self.train_set_x = None
        self.train_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self.n_train_batches = 0
        self.n_valid_batches = 0
        self.n_test_batches = 0
        self.cost = 0
        self.grads = None
        self.params = None
        self.output_layer = None
        self.index = 0
        self.input_shape = None
        self.x = None
        self.y = None

    def build_model(self):
        """ Creates the actual model from the model settings."""
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

        print 'Size train_batches %d, n_valid_batches %d, n_test_batches %d' % (self.n_train_batches, self.n_valid_batches, self.n_test_batches)

        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                               # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print 'Building the model ...'

        # The input is an 4D array of size, number of images in the batch size, number of channels
        # (or number of feature maps), image width and height.
        nbr_feature_maps = 1
        layer_input = self.x.reshape((self.batch_size, nbr_feature_maps, self.input_shape[0], self.input_shape[1]))
        pooled_width = self.input_shape[0]
        pooled_height = self.input_shape[1]
        # Add convolutional layers followed by pooling
        clayers = []
        for clayer_params in self.convolutional_layers:
            print 'Adding conv layer nbr filter %d, Ksize %d' % (clayer_params.num_filters, clayer_params.filter_w)
            layer = LeNetConvPoolLayer(self.rng, input=layer_input,
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_width, pooled_height),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps,
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize))
            clayers.append(layer)
            pooled_width = (pooled_width - clayer_params.filter_w + 1) / self.poolsize
            pooled_height = (pooled_height - clayer_params.filter_w + 1) / self.poolsize
            layer_input = layer.output
            nbr_feature_maps = clayer_params.num_filters


        # Flatten the output of the previous layers and add
        # fully connected sigmoidal layers
        layer_input = layer_input.flatten(2)
        nbr_input = nbr_feature_maps * pooled_width * pooled_height
        hlayers = []
        for hlayer_params in self.hidden_layers:
            print 'Adding hidden layer fully connected %d' % (hlayer_params.num_outputs)
            layer = HiddenLayer(self.rng, input=layer_input, n_in=nbr_input,
                                n_out=hlayer_params.num_outputs, activation=T.tanh)
            nbr_input = hlayer_params.num_outputs
            layer_input = layer.output
            hlayers.append(layer)

        # classify the values of the fully-connected sigmoidal layer
        self.output_layer = LogisticRegression(input=layer_input,
                                               n_in=nbr_input,
                                               n_out=self.last_layer.num_outputs)

        # the cost we minimize during training is the NLL of the model
        self.cost = self.output_layer.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.output_layer.params
        for hidden_layer in reversed(hlayers):
            self.params += hidden_layer.params
        for conv_layer in reversed(clayers):
            self.params += conv_layer.params


        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

    def train_model(self):
        """ Abstract method to be implemented by subclasses"""
        raise NotImplementedError()
