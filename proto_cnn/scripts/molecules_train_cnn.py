# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetresc
"""
import cPickle
import gzip
import numpy as np
import os
import scipy
import src.core
from train_cnn_regression_sq import CNNTrainRegressionSQ


class CNNTrainRegressionMolecules(CNNTrainRegressionSQ):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, cnn_settings_protofile, cached_weights_file, small_set=True):
        self.small_set = small_set
        super(CNNTrainRegressionSQ, self).__init__(cnn_settings_protofile, cached_weights_file)

    def load_samples(self):
        # Load datasets
        D = 500
        test_set = np.random.uniform(-1,0,(1000,D)), np.random.uniform(-1,0,1000)
        test_set_x, test_set_y = self.prepare_dataset(test_set)
        valid_set = np.random.uniform(-1,0,(500,D)), np.random.uniform(-1,0,500)
        valid_set_x, valid_set_y = self.prepare_dataset(valid_set)
        train_set = np.random.uniform(-1,0,(5000,D)), np.random.uniform(-1,0,5000)
        train_set_x, train_set_y = self.prepare_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                    (test_set_x, test_set_y)]
        return rval
 
    def prepare_dataset(self, dataset):
       """ Reshapes the input array to contain a dimension for
           the number of channels and made the set into shared variable"""
       x, y = dataset
       print y.shape
       x = np.reshape(x, (x.shape[0], x.shape[1])) 
       return self.shared_dataset((x, y))
