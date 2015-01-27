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

from google.protobuf import text_format

from test_tp_cnn import CNNTestTP
from lenet_conv_pool_layer import LeNetConvPoolLayer
from lenet_layer_conv_pool_non_symbolic import LeNetLayerConvPoolNonSymbolic
from lenet_layer_conv_pool_separable_non_symbolic import LeNetLayerConvPoolSeparableNonSymbolic
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

class CNNTestTPimagenet(CNNTestTP):
    def __init__(self, protofile, cached_weights_file, frame = None):
		self.frame = frame
		super(CNNTestTPmnist, self).__init__(protofile, cached_weights_file)

    def load_samples(self):
        img = np.load('../data/img11.npy') )
        test_set_x, test_set_y = shared_dataset(test_set)
    	rval = (test_set_x, test_set_y)
    	return rval
