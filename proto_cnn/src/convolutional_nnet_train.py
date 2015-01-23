# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vivianapetrescu
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
from load_mitocondria import load_mitocondria
from load_data_rescaled import load_mnist_data_rescaled
from load_data import load_mnist_data
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


class ConvolutionalNeuralNetworkTrain(object):
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
        logger_filename = file_path
        d = datetime.datetime.now()
        # add calendar day information
        logger_filename += '_' + str(d.day) + '_' + str(d.month) + '_' + str(d.year);
        # add hour information
        logger_filename += '_' + str(d.hour) + '_' + str(d.minute) + '_' + str(d.second);
        # add extension
        logger_filename += '.log'
        logging.basicConfig(filename=logger_filename, level=logging.INFO)


    def build_model(self):
	# Fixed rng, make the results repeatable
        rng = numpy.random.RandomState(23455)

        datasets = load_mitocondria()

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

        # The input is an 4D array of size, number of images in the batch size, number of channels
	# (or number of feature maps), image width and height.
	nbr_feature_maps = 1
	layer_input = self.x.reshape((self.batch_size, nbr_feature_maps, self.input_shape[0], self.input_shape[1]))
        pooled_W = self.input_shape[0];
        pooled_H = self.input_shape[1];
	# Add convolutional layers followed by pooling
        clayers = []
        for clayer_params in self.convolutional_layers:
            print 'Adding conv layer nbr filter %d, Ksize %d' % (clayer_params.num_filters, clayer_params.filter_w)
            layer = LeNetConvPoolLayer(rng, input = layer_input,
                                       image_shape=(self.batch_size, nbr_feature_maps, pooled_W, pooled_H),
                                       filter_shape=(clayer_params.num_filters, nbr_feature_maps,
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize))
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
            layer = HiddenLayer(rng, input=layer_input, n_in=nbr_input,
                         n_out = hlayer_params.num_outputs, activation=T.tanh)
            nbr_input = hlayer_params.num_outputs;
            layer_input = layer.output
            hlayers.append(layer)

        # classify the values of the fully-connected sigmoidal layer
        self.output_layer = LogisticRegression(input=layer_input, n_in= nbr_input, n_out = self.output_layer.num_outputs)

        # the cost we minimize during training is the NLL of the model
        self.cost = self.output_layer.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.output_layer.params
        for hl in reversed(hlayers):
            self.params += hl.params
        for cl in reversed(clayers):
            self.params += cl.params


        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)


    def train_model(self):
          # train_model is a function that updates the model parameters by
          # SGD Since this model has many parameters, it would be tedious to
          # manually create an update rule for each model parameter. We thus
          # create the updates list by automatically looping over all
          # (params[i],grads[i]) pairs.
          updates = []
          for param_i, grad_i in zip(self.params, self.grads):
              updates.append((param_i, param_i - self.learning_rate * grad_i))

          train_model = theano.function([self.index], [self.cost, self.output_layer.VOC_values(self.y)], updates=updates,
              givens={
                self.x: self.train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]},
                name = 'train_model')

             # create a function to compute the mistakes that are made by the model
          self.test_model = theano.function([self.index], [self.output_layer.y_pred, self.output_layer.VOC_values(self.y)],
             givens={
                self.x: self.test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]},
                name = 'test_model')
          self.validate_model = theano.function([self.index], [self.output_layer.y_pred, self.output_layer.VOC_values(self.y)],
            givens={
                self.x: self.valid_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.valid_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]},
                name = 'validate_model')
            ###############
            # TRAIN MODEL #
            ###############
          print '... training'
           # early-stopping parameters
          patience = 1000000 # look as this many examples regardless
          patience_increase = 2  # wait this much longer when a new best is
                           # found
          improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
          validation_frequency = min(self.n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

          self.best_params = None
          best_validation_loss = 0#numpy.inf
          test_score = 0.

          epoch = 0
          done_looping = False

          mean_training_time = 0
          cnt_times = 0
          while (epoch < self.n_epochs) and (not done_looping):
              epoch = epoch + 1
              for minibatch_index in xrange(self.n_train_batches):
		  # The model will process the iter batch
                  iter = (epoch - 1) * self.n_train_batches + minibatch_index

                  if iter % 100 == 0:
                      print 'training @ iter = ', iter
                  start = time.time()
                  [train_cost, train_voc_values] = train_model(minibatch_index)
                  end = time.time()
                  mean_training_time += end - start
                  cnt_times += 1
		  logging.info('cost %f, VOC %f' % (train_cost, train_voc_values))

                  if (iter + 1) % 1000 == 0: #validation_frequency == 0:

                    # compute zero-one loss on validation set
                    this_validation_loss = self.compute_validation_VOC_loss() 
                    logging.info('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, self.n_train_batches, \
                       this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss > best_validation_loss:

                      #improve patience if loss improvement is good enough
                      if this_validation_loss > best_validation_loss:
                          patience = max(patience, iter * patience_increase)

                          # save best validation score and iteration number
                          best_validation_loss = this_validation_loss
                          self.best_params = self.params

                          # test it on the test set
                          test_score = self.compute_test_VOC_loss()
                          logging.info(('     epoch %i, minibatch %i/%i, test error of best '
                             'model %f %%') %
                             (epoch, minibatch_index + 1, self.n_train_batches,
                             test_score * 100.))

                    if patience <= iter:
                        done_looping = True
                        break
          print 'Saving best parameters'
          self.save_parameters()
          mean_training_time /= cnt_times
          print 'running_times %f', mean_training_time
	  logging.info(('running time %f' % (mean_training_time)))
    def compute_validation_VOC_loss(self):
	  # works for 0-1 loss
	  all_y_pred = numpy.empty([])
	  for i in xrange(self.n_valid_batches):
		[y_pred, validation_loss] = self.validate_model(i)
		if i == 0:
			all_y_pred = y_pred
		else:
         	        all_y_pred = numpy.concatenate((all_y_pred, y_pred))
	  print all_y_pred

          F = T.sum(T.neq(self.valid_set_y, all_y_pred))
          TP = T.sum(T.and_(T.eq(self.valid_set_y, 1), T.eq(all_y_pred, 1)))
	  result =  TP/T.cast(TP+F, theano.config.floatX)
          print 'Print result is ', result.eval()
	  return result.eval() 

    def compute_test_VOC_loss(self):
	  # works for 0-1 loss
          all_y_pred = numpy.empty([])
	  for i in xrange(self.n_test_batches):
		[y_pred, test_loss] = self.test_model(i)
		if i==0:
		    all_y_pred = y_pred
		else:
         	    all_y_pred = numpy.concatenate((all_y_pred, y_pred))
          print all_y_pred
	  print all_y_pred.shape
	  F = T.sum(T.neq(self.test_set_y, all_y_pred))
          TP = T.sum(T.and_(T.eq(self.test_set_y, 1), T.eq(all_y_pred, 1)))
	  result =  TP/T.cast(TP+F, theano.config.floatX)
          print 'Print result is ', result.eval()
	  return result.eval() 

    def save_parameters(self):
          weights = [i.get_value(borrow=True) for i in self.best_params]
          numpy.save(self.cached_weights_file, weights)


