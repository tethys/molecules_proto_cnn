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
class ConvolutionalNeuralNetworkTrain(object):
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
        # default values        
        self.learning_rate = 0.9;
        self.n_epochs = 1;
        # This field is required
        self.dataset = settings.dataset;
        self.batch_size = 100;
        self.poolsize = 5;
        # TODO this
        self.convolutional_layers = [];
        self.hidden_layers = [];
        for layer in settings.conv_layer:
              self.convolutional_layers.append(layer)
        for layer in settings.hidden_layer:
              self.hidden_layers.append(layer)      

        # required at least one layer
        self.output_layer = settings.last_layer;


        # required parameter
        self.cost_function = settings.cost_function;
        
        if settings.HasField('learning_rate'):
               self.learning_rate = settings.learning_rate;
        if settings.HasField('n_epochs'):
                self.n_epochs = settings.n_epochs;        
        if settings.HasField('batch_size'):
                self.batch_size = settings.batch_size;
        if settings.HasField('poolsize'):
                self.poolsize = settings.poolsize;

        self.input_shape = (28,28); # this is the size of MNIST images

    def build_model(self):
        rng = numpy.random.RandomState(23455)

        datasets = load_mnist_data(self.dataset)

        # Train, Validation, Test 50000, 10000, 10000 times 28x28 = 784
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        # 50000/50 = 1000, 10000/50 = 200
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= self.batch_size
        self.n_valid_batches /= self.batch_size
        self.n_test_batches /= self.batch_size

        print('Size train ',self.train_set_x.shape.eval() , ', valid ' , self.valid_set_x.shape.eval(), ', test ' , self.test_set_x.shape.eval())
        print('Size train_batches %d, n_valid_batches %d, n_test_batches %d' %(self.n_train_batches, self.n_valid_batches, self.n_test_batches))

        self.n_epochs = 1;
        # allocate symbolic variables for the data
        self.index = T.lscalar()  # index to a [mini]batch
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print 'Building the model ...'
        nkerns = [2,5]
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        layer0_input = self.x.reshape((self.batch_size, 1, self.input_shape[0], self.input_shape[1]))
        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(self.batch_size, 1, 28, 28),
            filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)

        # layer0 output is a 4d tensor as well
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
              image_shape=(self.batch_size, nkerns[0], 12, 12),
              filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        layer2_input = layer1.output.flatten(2)
        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * 4 * 4,
                         n_out=500, activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

        # the cost we minimize during training is the NLL of the model
        self.cost = self.layer3.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + layer2.params + layer1.params + layer0.params

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

          train_model = theano.function([self.index], self.cost, updates=updates,
              givens={
                self.x: self.train_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.train_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]})

             # create a function to compute the mistakes that are made by the model
          self.test_model = theano.function([self.index], self.layer3.errors(self.y),
             givens={
                self.x: self.test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]})
          self.validate_model = theano.function([self.index], self.layer3.errors(self.y),
            givens={
                self.x: self.valid_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                self.y: self.valid_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]})
            ###############
            # TRAIN MODEL #
            ###############
          print '... training'
           # early-stopping parameters
          patience = 10000  # look as this many examples regardless
          patience_increase = 2  # wait this much longer when a new best is
                           # found
          improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
          validation_frequency = min(self.n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

          best_params = None
          best_validation_loss = numpy.inf
          best_iter = 0
          test_score = 0.
          start_time = time.clock()

          epoch = 0
          done_looping = False

          while (epoch < self.n_epochs) and (not done_looping):
              epoch = epoch + 1
              for minibatch_index in xrange(self.n_train_batches):

                  iter = (epoch - 1) * self.n_train_batches + minibatch_index

                  if iter % 100 == 0:
                      print 'training @ iter = ', iter
                  cost_ij = train_model(minibatch_index)

                  if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                      (epoch, minibatch_index + 1, self.n_train_batches, \
                       this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                      #improve patience if loss improvement is good enough
                      if this_validation_loss < best_validation_loss * improvement_threshold:
                          patience = max(patience, iter * patience_increase)

                          # save best validation score and iteration number
                          best_validation_loss = this_validation_loss
                          best_iter = iter
                          self.best_params = self.params
                      
                          # test it on the test set
                          test_losses = [self.test_model(i) for i in xrange(self.n_test_batches)]
                          test_score = numpy.mean(test_losses)
                          print(('     epoch %i, minibatch %i/%i, test error of best '
                             'model %f %%') %
                             (epoch, minibatch_index + 1, self.n_train_batches,
                             test_score * 100.))

                    if patience <= iter:
                        done_looping = True
                        break
          print 'Saving best parameters'
          self.save_parameters()


    def save_parameters(self):
            save_file = open('cnn_model_original.pkl','wb')
            cPickle.dump(self.best_params, save_file, -1)
            save_file.close()


        
    
        
    
    
    
    
