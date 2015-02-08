# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetrescu
"""

import logging
import numpy
import theano
import theano.tensor as T
import time
from retrain_cnn import CNNRetrain

class CNNRetrainTP(CNNRetrain):
    """The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, protofile, cached_weights_file):
        super(CNNRetrainTP, self).__init__(protofile, cached_weights_file)
        self.test_model = None
        self.validate_model = None

    def retrain_model(self):
        """ """
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
                    name='train_model')

             # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function([self.index], [self.output_layer.y_pred, self.output_layer.VOC_values(self.y)],
                        givens={
                        self.x: self.test_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                        self.y: self.test_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]},
                        name='test_model')
        self.validate_model = theano.function([self.index], [self.output_layer.y_pred, self.output_layer.VOC_values(self.y)],
                            givens={
                            self.x: self.valid_set_x[self.index * self.batch_size: (self.index + 1) * self.batch_size],
                            self.y: self.valid_set_y[self.index * self.batch_size: (self.index + 1) * self.batch_size]},
                            name='validate_model')
            ###############
            # TRAIN MODEL #
            ###############
        print '... Retraining'
           # early-stopping parameters
        patience = 1000000 # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                           # found

        self.best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        epoch = 0
        mean_training_time = 0
        cnt_times = 0
        while (epoch < self.n_epochs):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):
            # The model will process the iter batch
                iteration = (epoch - 1) * self.n_train_batches + minibatch_index

                if iteration % 100 == 0:
                    print 'training @ iter = ', iteration
                start = time.time()
                [train_cost, train_voc_values] = train_model(minibatch_index)
                end = time.time()
                mean_training_time += end - start
                cnt_times += 1
                logging.info('cost %f, VOC %f' % (train_cost, train_voc_values))

                if (iteration + 1) % 1000 == 0: #validation_frequency == 0:

                    # compute zero-one loss on validation set
                    this_validation_loss = self.compute_validation_loss() 
                    logging.info('epoch %i, minibatch %i/%i, validation error %f %%' % \
                        (epoch, minibatch_index + 1, self.n_train_batches, \
                         this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        self.best_params = self.params
                        self.save_parameters()
                        # test it on the test set
                        test_score = self.compute_test_loss()
                        logging.info(('     epoch %i, minibatch %i/%i, test error of best '
                             'model %f %%') %
                             (epoch, minibatch_index + 1, self.n_train_batches,
                             test_score * 100.))

        print 'Saving best parameters'
        self.save_parameters()
        mean_training_time /= cnt_times
        print 'running_times %f', mean_training_time
        logging.info(('running time %f' % (mean_training_time)))

    def compute_validation_loss(self):
        """Computes  error loss"""
        # works for 0-1 loss
        all_y_pred = numpy.empty([])
        for i in xrange(self.n_valid_batches):
            y_pred = self.validate_model(i)[0]
            if i == 0:
                all_y_pred = y_pred
            else:
                all_y_pred = numpy.concatenate((all_y_pred, y_pred))
            print all_y_pred
        result = T.mean(T.neq(self.test_set_y, all_y_pred))
        print 'Print result is ', result.eval()
        return 1.0 - result.eval()

    def compute_test_loss(self):
        """Computes error loss"""
        # works for 0-1 loss
        all_y_pred = numpy.empty([])
        for i in xrange(self.n_test_batches):
            y_pred = self.test_model(i)[0]
            if i == 0:
                all_y_pred = y_pred
            else:
                all_y_pred = numpy.concatenate((all_y_pred, y_pred))
        print all_y_pred
        print all_y_pred.shape
        result = T.mean(T.neq(self.test_set_y, all_y_pred))
        print 'Print result is ', result.eval()
        return 1.0 - result.eval()

