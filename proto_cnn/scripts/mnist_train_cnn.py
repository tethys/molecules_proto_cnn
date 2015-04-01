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
from src.core.train_tp_cnn import CNNTrainTP


class CNNTrainTPmnist(CNNTrainTP):
    """ The class takes a proto bufer as input, setups a CNN according to the
        settings, trains the network and saves the weights in a file
    """
    def __init__(self, cnn_settings_protofile, cached_weights_file, small_set=True):
        self.small_set = small_set
        super(CNNTrainTPmnist, self).__init__(cnn_settings_protofile, cached_weights_file)

    def load_samples(self):
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
        train_set, valid_set, test_set = cPickle.load(f)
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
            N = 10000
            tmp_images = np.zeros((N, 56, 56))
            for i in range(N):
                tmp_images[i, :, :] = scipy.misc.imresize(valid_set[0][i, :, :], (56, 56), interp='nearest')
            valid_set_x, valid_set_y = self.shared_dataset((tmp_images, valid_set[1]))
            N = 60000
            tmp_images = np.zeros((N, 56, 56))
            for i in range(N):
                tmp_images[i, :, :] = scipy.misc.imresize(train_set[0][i, :, :], (56, 56), interp='nearest')
            train_set_x, train_set_y = self.shared_dataset((tmp_images, train_set[1]))
        else:
            test_set_x, test_set_y = self.prepare_dataset(test_set)
            valid_set_x, valid_set_y = self.prepare_dataset(valid_set)
            train_set_x, train_set_y = self.prepare_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                    (test_set_x, test_set_y)]
        return rval
   
    def prepare_dataset(self, dataset):
       """ Reshapes the input array to contain a dimension for
           the number of channels and made the set into shared variable"""
       x, y = dataset
       print 'y shape ', y.shape
       x = np.reshape(x, (x.shape[0], x.shape[1])) 
       return self.shared_dataset((x, y))
