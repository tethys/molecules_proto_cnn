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
import os
import os.path
import sys
import time
import multiprocessing as mp

import numpy
import numpy.linalg
import numpy as np
from numpy import *
from numpy.random import *
#from cvutils import patches
#from cvutils import misc
import patches
import misc
from scipy.misc import imsave

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None):
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
            W = theano.shared(numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),
                                   borrow=True)
            
        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)

        self.W = W
        self.b = b
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def shared_dataset(data_x,data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
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


def imsave_file(CNN,
                num_epoch,
                frame,
                nkerns=[10, 20, 50],
                batch_size=661,
                training_path=None,
                output_path=None):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    
    print "Processing slice %s..." % frame
    
    if training_path is None:
        training_path = CNN
    
    if output_path is None:
        output_path = CNN
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    theano.config.floatX = 'float32'
    rng = numpy.random.RandomState(23455)

    areasize = 51
    gap = (areasize - 1)/2
    
    # Extract all 51x51 patches from testing data
    centers = patches.grid_patches_centers(Vtest[frame,:,:].shape, (areasize,areasize))
    test_set_x = patches.get_many_patches(Vtest[frame,:,:], (areasize,areasize), centers, flat=True)
    test_set_y = Gtest[frame][centers[:,0], centers[:,1]]

    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    
    # compute number of minibatches
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_test_batches /= batch_size
    
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
    # Load weights...
    weights = numpy.load(os.path.join(training_path, 'weights_'+CNN+'_epoch'+str(num_epoch)+'.npy'))
    
    W3 = theano.shared(weights[0])
    b3 = theano.shared(weights[1])
    W2 = theano.shared(weights[2])
    b2 = theano.shared(weights[3])
    W1 = theano.shared(weights[4])
    b1 = theano.shared(weights[5])
    W0b = theano.shared(weights[6])
    b0b = theano.shared(weights[7])
    W0 = theano.shared(weights[8])
    b0 = theano.shared(weights[9])
    
    # Create the CNN with loaded weights
    layer0_input = x.reshape((batch_size, 1, 51, 51))

    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 51, 51),
            filter_shape=(nkerns[0], 1, 6, 6), poolsize=(2, 2),W=W0,b=b0)
    
    layer0b = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 23, 23),
            filter_shape=(nkerns[1], nkerns[0], 6, 6), poolsize=(2, 2),W=W0b,b=b0b)

    layer1 = LeNetConvPoolLayer(rng, input=layer0b.output,
            image_shape=(batch_size, nkerns[1], 9, 9),
            filter_shape=(nkerns[2], nkerns[1], 6, 6), poolsize=(2, 2),W=W1,b=b1)
    
    layer2_input = layer1.output.flatten(2)
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[2] * 2 * 2,
                         n_out=500, activation=T.tanh,W=W2,b=b2)

    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=2,W=W3,b=b3)
    
    # Prediction function
    output_pred_model = theano.function([index], layer3.p_y_given_x,
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]})
    
    # Prediction in minibatches
    outputs = numpy.hstack([output_pred_model(i)[:,1] for i in xrange(n_test_batches)])    #for the result of the sigmoid (1 if mito in white)
    outputs = outputs.reshape(422-2*gap,711-2*gap)
    
    # Save the results as images and npy
    out_filename =  os.path.join(output_path,'%s_train_epoch%s_frame%i.png' % (CNN, num_epoch, frame+1))
    imsave(out_filename, outputs)
    out_filename =  os.path.join(output_path,'%s_train_epoch%s_frame%i' % (CNN, num_epoch, frame+1))
    np.save(out_filename, outputs)

def __imsave_file(frame):
    imsave_file(CNN='CNN55', num_epoch='0150', frame=frame,
        training_path="CNN55", output_path="output")

if __name__ == '__main__':
    # Load data
    Vtest = float32(misc.tiffread('Volume_test.tif')/255.0)
    Gtest = uint8(misc.tiffread('Ground_truth_test.tif'))
    Gtest[Gtest==255] = 1
    
    # Process
    # for i in xrange(0,318):
    #     imsave_file(CNN='CNN55',num_epoch='0150',frame = i,
    #         training_path="CNN55", output_path="output")
    
    pool = mp.Pool(4)
    pool.map(__imsave_file, xrange(318))
