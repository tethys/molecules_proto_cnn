# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 15:27:51 2014
@author: vpetresc
"""

from base_cnn import CNNBase
from lenet_layer_conv_pool_non_symbolic import LeNetLayerConvPoolNonSymbolic
from lenet_layer_conv_pool_separable_non_symbolic import LeNetLayerConvPoolSeparableNonSymbolic
from lenet_conv_pool_layer import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
import numpy
import time
import theano
import theano.tensor as T


class CNNRetrain(CNNBase):
    """ The class is responsbile for retraining the weights of the CNN. The class retrains
        the biases and the last weights of the network.  """
    def __init__(self, protofile, cached_weights):
        self.cnntype = 'RETRAIN'
        super(CNNRetrain, self).__init__(protofile, cached_weights)
        #: Overloaded by derived classes
        self.load_weights()
        self.train_set_x = None
        self.train_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.input_shape = None
        self.best_params = None

    def build_model(self):
        """Create the actual model"""
        datasets = self.load_samples()
        # Train, Validation, Test 100000, 20000, 26... fot Mitocondria set
        # Train, Validation, Test 50000, 10000, 10000 times 28x28 = 784 for MNIST dataset
	train_set_x, self.train_set_y = datasets[0]
        valid_set_x, self.valid_set_y = datasets[1]
        test_set_x, self.test_set_y = datasets[2]
	self.train_set_x = train_set_x.get_value()
	self.valid_set_x = valid_set_x.get_value()
	self.test_set_x = test_set_x.get_value()

        # assumes the width equals the height
        img_width_size = numpy.sqrt(self.test_set_x.shape[1]).astype(int)
        print "Image shape %s x %s" % (img_width_size, img_width_size)
        self.input_shape = (img_width_size, img_width_size)

        # Compute number of minibatches for training, validation and testing
        # Divide the total number of elements in the set by the batch size
        self.n_train_batches = self.train_set_x.shape[0]
        self.n_valid_batches = self.valid_set_x.shape[0]
        self.n_test_batches = self.test_set_x.shape[0]
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
        #for i in xrange(len(self.cached_weights)):
        #    self.cached_weights[i] = theano.shared(self.cached_weights[i])
        # The input is an 4D array of size, number of images in the batch size, number of channels
        # (or number of feature maps), image width and height.
        nbr_feature_maps = 1
        layer_input = self.x.reshape((self.batch_size, nbr_feature_maps, self.input_shape[0], self.input_shape[1]))
        pooled_width = self.input_shape[0];
        pooled_height = self.input_shape[1];
        # Add convolutional layers followed by pooling
        clayers = []
        idx_weight = 0
        for clayer_params in self.convolutional_layers:
            print 'Adding conv layer nbr filter %d, kernel size %d' % (clayer_params.num_filters, clayer_params.filter_w)
            if clayer_params.HasField('rank') == False:
                layer_conv = LeNetConvPoolNonSymbolic(self.rng)
                layer_output = layer_convolutoinal.run_batch(layer_input,
                                                    image_shape=(self.batch_size, nbr_feature_maps,
                                                    pooled_width, pooled_height),
                                                    filter_shape=(clayer_params.num_filters, nbr_feature_maps,
                                                            clayer_params.filter_w, clayer_params.filter_w),
                                                    W=self.cached_weights[idx_weight + 1],
                                                    b=self.cached_weights[idx_weight],
                                                    poolsize=(self.poolsize, self.poolsize)).eval()
            else:
                layer_sep_conv = LeNetLayerConvPoolSeparableNonSymbolic(self.rng)
                print nbr_feature_maps
                print pooled_width, pooled_height
                layer_output = layer_sep_conv.run_batch(layer_input,
                                                        image_shape=(self.batch_size, nbr_feature_maps,
                                                                pooled_width, pooled_height),
                                                        filter_shape=(clayer_params.num_filters, nbr_feature_maps,
                                                                clayer_params.filter_w, clayer_params.filter_w),
                                                        Pstruct=self.cached_weights[idx_weight+1],
                                                        b=self.cached_weights[idx_weight],
                                                        poolsize=(self.poolsize, self.poolsize))
            pooled_width = (pooled_width - clayer_params.filter_w + 1) / self.poolsize
            pooled_height = (pooled_height - clayer_params.filter_w + 1) / self.poolsize
            layer_input = layer_output
            nbr_feature_maps = clayer_params.num_filters
            idx_weight += 2

        # fully connected sigmoidal layers
        layer_input = layer_input.flatten(2);
        nbr_input = nbr_feature_maps * pooled_W * pooled_H
        hlayers = []
        for hlayer_params in self.hidden_layers:
            print 'Adding hidden layer fully connected %d' % (hlayer_params.num_outputs)
            layer = HiddenLayer(self.rng, input=layer_input, n_in=nbr_input,
                         n_out=hlayer_params.num_outputs, activation=T.tanh,
                         W=self.cached_weights[idx_weight +1], b=self.cached_weights[idx_weight])
            nbr_input = hlayer_params.num_outputs;
            layer_input = layer.output
            hlayers.append(layer)
            idx_weight += 2

        # classify the values of the fully-connected sigmoidal layer
        self.output_layer = LogisticRegression(input=layer_input, n_in= nbr_input,
                        n_out = self.last_layer.num_outputs, W=self.cached_weights[idx_weight + 1],
                        b=self.cached_weights[idx_weight])

        # the cost we minimize during training is the NLL of the model
        self.cost = self.output_layer.negative_log_likelihood(self.y)

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.output_layer.params
        for hidden_layer in reversed(hlayers):
            self.params += hidden_layer.params
        for conv_layer in reversed(clayers):
            self.params += conv_layer.b_params

        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

    def retrain_model(self):
        """Abstract method"""
        raise NotImplementedError()

    def save_parameters(self):
        """Save the retrained weights"""
        weights = [i.get_value(borrow=True) for i in self.best_params]
        ## add here the interleaved convolutional layers
        nbr_hidden_layers = size(hlayers)
        # Update the output layer and W,b for every hidden layer
        toskip = 1 + nbr_hidden_layers * 2
        retrained_weights = []
        for widx in xrange(toskip):
            retrained_weights.append(weights[widx])
        for c in xrange(size(clayers)):
            # Add old W weights for conv layers(corresponding to the
            # filters that were not learned)
            retrained_weights.append(self.cached_weights[2*c + 1])
            # Add b for conv layer that was trained
            retrained_weights.append(weights[toskip])
            toskip += 1
        file_path = os.path.splitext(self.cached_weights_file)[0]
        numpy.save(file_path +'_retrain.npy', retrained_weights)
