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

from test_cnn import CNNTest
from lenet_conv_pool_layer import LeNetConvPoolLayer
from lenet_layer_conv_pool_non_symbolic import LeNetLayerConvPoolNonSymbolic
from lenet_layer_conv_pool_separable_non_symbolic import LeNetLayerConvPoolSeparableNonSymbolic
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

class CNNTestVOC(CNNTest):

    def compute_batch_error(self, batch_result_dict):
	TP = batch_result_dict['TP']
	FP = batch_result_dict['FP']
	FN = batch_result_dict['FN']
        return TP/float(TP + FP + FN) 
    def compute_all_samples_error(self, all_samples_result):
	TP = 0
	FP = 0
	FN = 0
	for result in all_samples_result:
		TP += result['TP']
		FP += result['FP']
		FN += result['FN']
	return TP/float(TP + FP + FN)
