# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:39:22 2014

@author: vivianapetrescu
"""

import cPickle
import gzip
import numpy as np
import os
import scipy

from src.core.test_tp_cnn import CNNTestTP

class CNNTestTPmnist(CNNTestTP):
    """ Loads the MNIST data set"""
    def __init__(self, protofile, cached_weights_file, small_set=True):
        self.small_set = small_set
        super(CNNTestTPmnist, self).__init__(protofile, cached_weights_file)

    def load_samples(self):
        """ Loads the normal sised images or the upscaled version"""
        print 'Value of small set is ', self.small_set
        # Load datasets
        dataset = 'mnist.pkl.gz'
        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)

        print '... loading data'

        # Load the dataset
        f = gzip.open(dataset, 'rb')
        test_set = cPickle.load(f)[2]
        f.close()
        #train_set, valid_set, test_set format: tuple(input, target)
        #input is an numpy.ndarray of 2 dimensions (a matrix)
        #witch row's correspond to an example. target is a
        if self.small_set == False:
        # Upscale the data
            N = 10000
            tmp_images = np.zeros((N, 56, 56))
            for i in range(N):
                tmp_images[i, :, :] = scipy.misc.imresize(test_set[0][i, :, :], (56, 56))
            test_set_2 = (tmp_images, test_set[1])
            test_set_x, test_set_y = self.shared_dataset(test_set_2)
        else:
            test_set_x, test_set_y = self.shared_dataset(test_set)
        return (test_set_x, test_set_y)
