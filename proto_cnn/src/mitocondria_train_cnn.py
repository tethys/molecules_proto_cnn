# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetresc
"""

import numpy as np
from train_voc_cnn import CNNTrainVOC

class CNNTrainVOCMitocondria(CNNTrainVOC):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, cnn_settings_protofile, cached_weights_file, small_set):
        self.small_set = small_set
        super(CNNTrainVOCMitocondria, self).__init__(cnn_settings_protofile, cached_weights_file)

    def load_samples(self):
        """ Load mitocondria data """
        print 'Load Mitocondria small Set ', self.small_set
        # Load datasets
        path_to_data = '../data/'
        if self.small_set == True:
            train_set_x = np.load(path_to_data + 'train_set_x_100000_51.npy')
            train_set_y = np.load(path_to_data + 'train_set_y_100000_51.npy')
            valid_set_x = np.load(path_to_data + 'valid_set_x_20000_51.npy')
            valid_set_y = np.load(path_to_data +'valid_set_y_20000_51.npy')
        else:
            train_set_x = np.load(path_to_data + 'train_set_x_1000000_51.npy')
            train_set_y = np.load(path_to_data + 'train_set_y_1000000_51.npy')
            valid_set_x = np.load(path_to_data + 'valid_set_x_200000_51.npy')
            valid_set_y = np.load(path_to_data + 'valid_set_y_200000_51.npy')

        test_set_x = np.load(path_to_data +'test_set_x_fr1_51.npy')
        test_set_y = np.load(path_to_data +'test_set_y_fr1_51.npy')

        train_set_x, train_set_y = self.shared_dataset((train_set_x, train_set_y))
        valid_set_x, valid_set_y = self.shared_dataset((valid_set_x, valid_set_y))
        test_set_x, test_set_y = self.shared_dataset((test_set_x, test_set_y))
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval
