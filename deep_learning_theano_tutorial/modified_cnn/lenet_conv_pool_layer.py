"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import scipy.io
import numpy
import time

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2),  W=None, b=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        if W is None:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                               borrow=True)
        else:
            self.W = W

        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self.b = b

        print self.W.shape
        self.run_conv_pool(filter_shape, image_shape, poolsize)
       # self.run_separable_conv_pool(filter_shape, image_shape, poolsize)
        
        
    def run_conv_pool(self, filter_shape, image_shape, poolsize):    
        # convolve input feature maps with filters
        start = time.time()
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        
        end = time.time()
        self.convolutional_time = (end - start)*1000/image_shape[0]                
        
        # downsample each feature map individually, using maxpooling
        start = time.time()
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        end = time.time()
        self.downsample_time = (end - start)*1000/ image_shape[0]
        
        print 'conv {0}, {1} ms'.format(self.convolutional_time, self.downsample_time)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        print 'pooled out shape ', pooled_out.shape
        # store parameters of this layer
        self.params = [self.W, self.b]

    def run_separable_conv_pool(self, filter_shape, image_shape, poolsize):    
        # convolve input feature maps with filters
        start = time.time()
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        
        end = time.time()
        self.convolutional_time = (end - start)*1000/image_shape[0]
        

        pWeights = scipy.io.loadmat('./experiments/setup_theano_tutorial/cnn_separable_model.mat')        
        val = pWeights['P']
        av = val[0,0]
        sep_filters = av[1][2];
        print sep_filters.shape
        print sep_filters[0]
        w_tensors = T.as_tensor_variable(sep_filters[0]);
        input_4D = w_tensors.reshape((10,1,1,5))
        print input_4D
        if filter_shape == (20,1,5,5):
            conv_outx = conv.conv2d(input=input, filters=input_4D,
                                    filter_shape=(10,1,1,5), image_shape=image_shape)
            print 'Running!'
                
        
        # downsample each feature map individually, using maxpooling
        start = time.time()
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)
        end = time.time()
        self.downsample_time = (end - start)*1000/ image_shape[0]
        
        print 'conv {0}, {1} ms'.format(self.convolutional_time, self.downsample_time)
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        print 'pooled out shape ', pooled_out.shape
        # store parameters of this layer
        self.params = [self.W, self.b]

