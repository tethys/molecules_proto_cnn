# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetresc

"""

import datetime
import logging
import numpy as np
import os

import theano
import theano.tensor as T

import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format


class CNNBase(object):
    """ The class sets up its internal layers configuration
        based on an input .prototxt and the name of the
        weights file
    """

    def __init__(self, cnn_settings_protofile, cached_weights_file):
        """  The method parses the proto file and saves the
             filename of the weights file.
        :param cnn_settings_protofile: The .prototxt file that contains
                                        the network configuration
        :type cnn_settings_protofile: pb_cnn google protocol buffer
        :param cached_weights_file: filename  of the weights file (either for
                                    loading or for saving)
        :type cached_weights_file: string

        :raises: IOError
        """
        self.cached_weights_file = cached_weights_file
        # Will hold the weights (including biases) for all layers)
        self.cached_weights = []
        # Initalizes the logger name and debug level similar.
        self.initialize_logger()
        # Random generator used for initializing the weights 
        # in case of training and retraining.
        self.rng = np.random.RandomState(23455)
        # Create protobuff empty object
        settings = pb_cnn.CNNSettings()
        try:
            proto_file = open(cnn_settings_protofile, "r")
            data = proto_file.read()
            text_format.Merge(data, settings)
            print "Network settings are \n %s \n", data
            logging.debug(data)
            self.create_layers_from_settings(settings)
            proto_file.close()
        except IOError:
            print "Could not open file " + cnn_settings_protofile
        # Default values for parameters in case they are not
        # provided in the prototxt file.
        #: Default learning rate for stochastic GD
        self.learning_rate = 0.1
        #: Default nbr of epochs
        self.n_epochs = 100
        #: Default batch size
        self.batch_size = 100
        #: Default poolsize
        self.poolsize = 2

    def create_layers_from_settings(self, settings):
        """Takes as input the net settings parsed from the proto file
            and sets up the CNN configuration.

        :param settings: network settings (e.g. layers type, count, poolsize,
                                            batch size, learning rate)
        :type settings: protocol buffer

        """
        if settings.HasField('learning_rate'):
            self.learning_rate = settings.learning_rate
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
            self.poolsize = settings.poolsize
        else:
            print 'Warning - default number of poolsize ', self.poolsize

        # Add every convolutional and hidden layer to an array.
        self.convolutional_layers = []
        self.hidden_layers = []
        for layer in settings.conv_layer:
            self.convolutional_layers.append(layer)
        for layer in settings.hidden_layer:
            self.hidden_layers.append(layer)

        # Last layer type is required.
        self.last_layer = settings.last_layer

        # Required parameters TODO still needed?
        self.dataset = settings.dataset
        self.cost_function = settings.cost_function

    def initialize_logger(self):
        """Initializes the logging level and the logger
            name based on the path of the weights file.
        """
	## Remove old logger
	logger = logging.getLogger()
	if logger is not None:
	    for handler in logger.handlers[:]:
		logger.removeHandler(handler)
        # The log file is saved in the same folder with the cached weights file
        # but with a different extension.
        file_path = os.path.splitext(self.cached_weights_file)[0]
        current_time = datetime.datetime.now()
        # Add calendar day and hour information in the logfile name.
        logger_filename = "%s_%s_%d_%d_%d_%d_%d_%d.log" % (file_path, self.cnntype, 
                            current_time.day, current_time.month, current_time.year,
                            current_time.hour, current_time.minute, current_time.second)
        logging.basicConfig(filename=logger_filename, level=logging.DEBUG)

    def shared_dataset(self, data_xy, borrow=True):
        """Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.

        :param data_xy: tuple containing the input and output arrays
        :param borrow:  (Default value = True)

        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    def load_weights(self):
        """Load weights from .npy file. The weights are stored in reverse order.
           For a 4 layer network that would mean the contents of the weights are:
              b3 W3, b2 W2, b1 W1, b0 W0
          where b are the biases and W are the filters or the linear weights

        """
        all_weights = np.load(self.cached_weights_file)
        for weight in reversed(all_weights):
            self.cached_weights.append(weight)
            print 'weight array size ', len(weight)
        print 'Cached weights size is ', len(self.cached_weights)

    def load_samples(self):
        """Abstract method implemented by derived classes.
            Loads train, test and validation samples

        """
        raise NotImplementedError()
