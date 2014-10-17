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

import numpy
import numpy as np
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


def evaluate_lenet5(init_learning_rate=0.01, momentum_rate=0.6, decay_rate = 1,
                    n_epochs=2000, nkerns=[10, 20, 50], batch_size=400,
                    base_path=None):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    base_name = os.path.splitext(os.path.basename(__file__))[0]
    if base_path is None:
        base_path = base_name
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    theano.config.floatX = 'float32'
    rng = numpy.random.RandomState(23455)
    
    decay = np.float32(np.log(decay_rate)/n_epochs)
    gap = 25
    
    print "Loading data..."
    
    # Load datasets
    train_set_x = numpy.load('train_set_x_1000000_51.npy')
    train_set_y = numpy.load('train_set_y_1000000_51.npy')
    valid_set_x = numpy.load('valid_set_x_20000_51.npy')
    valid_set_y = numpy.load('valid_set_y_20000_51.npy')
    test_set_x = numpy.load('test_set_x_fr1_51.npy')
    test_set_y = numpy.load('test_set_y_fr1_51.npy')
    
    test_set_x, test_set_y = shared_dataset(test_set_x,test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x,valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x,train_set_y)
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    
    print "Building the model..."

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, 51, 51))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, 51, 51),
            filter_shape=(nkerns[0], 1, 6, 6), poolsize=(2, 2))
    
    layer0b = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], 23, 23),
            filter_shape=(nkerns[1], nkerns[0], 6, 6), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0b.output,
            image_shape=(batch_size, nkerns[1], 9, 9),
            filter_shape=(nkerns[2], nkerns[1], 6, 6), poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[2] * 2 * 2,
                         n_out=500, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=2)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.VOC(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    output_model = theano.function([index], layer3.y_pred,
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.VOC(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0b.params + layer0.params
    
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    
    # Learning rate depends on epoch
    epoch_theano = T.fscalar('epoch')
    learning_rate = init_learning_rate*T.exp(-decay*epoch_theano)
    
    updates = []
    old_delta = []
    for param_i, grad_i in zip(params, grads):
        # Previous parameter update for the momentum
        old_delta_i = theano.shared(np.zeros_like(param_i.get_value(borrow=True)))
        old_delta.append(old_delta_i)
        
        # New update is a linear combination of previous update and gradient (momentum)
        delta_i = - learning_rate * (1 - momentum_rate) * grad_i + momentum_rate*old_delta_i
        # Old delta should be current delta after update
        updates.append((old_delta_i, delta_i))
        # Parameter update
        updates.append((param_i, param_i + delta_i))
    
    learning_rate_fn = theano.function([epoch_theano], learning_rate)
    
    train_model = theano.function([index, epoch_theano], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    
    real_train_model = theano.function([index], cost,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    
    real_valid_model = theano.function([index], cost,
          givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    
    ###############
    # TRAIN MODEL #
    ###############
    print 'Training...'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = 0
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    f = open(os.path.join(base_path, base_name+'_train_cost_per_epoch.txt'),'w')
    g = open(os.path.join(base_path, base_name+'_valid_cost_per_epoch.txt'),'w')
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        print "\x1b[32m\x1b[1mStarting epoch %s...\x1b[0m" % epoch
        train_minibatch_costs = numpy.array([])
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            # if iter % 100 == 0:
            #     print 'training @ iter = ', iter
            
            cost_ij = train_model(minibatch_index, epoch)

            # if (iter + 1) % validation_frequency == 0:

            #     # compute zero-one loss on validation set
            #     validation_losses = [validate_model(i) for i
            #                          in xrange(n_valid_batches)]
            #     this_validation_loss = numpy.mean(validation_losses)
            #     print('epoch %i, minibatch %i/%i, validation VOC %f' % \
            #           (epoch, minibatch_index + 1, n_train_batches, \
            #            this_validation_loss))

            #     # if we got the best validation score until now
            #     if this_validation_loss > best_validation_loss:

            #         #improve patience if loss improvement is good enough
            #         if this_validation_loss > best_validation_loss /  \
            #            improvement_threshold:
            #             patience = max(patience, iter * patience_increase)

            #         # save best validation score and iteration number
            #         best_validation_loss = this_validation_loss
            #         best_iter = iter

            #         # test it on the test set
            #         test_losses = [test_model(i) for i in xrange(n_test_batches)]
            #         test_score = numpy.mean(test_losses)
            #         print(('     epoch %i, minibatch %i/%i, test VOC of best '
            #                'model %f') %
            #               (epoch, minibatch_index + 1, n_train_batches,
            #                test_score))

            # if patience <= iter:
            #     done_looping = True
            #     break
        train_losses = [real_train_model(i) for i in xrange(n_train_batches)]
        this_epoch_train_cost = numpy.mean(train_losses)
        valid_losses = [real_valid_model(i) for i in xrange(n_valid_batches)]
        this_epoch_valid_cost = numpy.mean(valid_losses)
        print >> f, this_epoch_train_cost
        print >> g, this_epoch_valid_cost
        f.flush()
        g.flush()
        print 'this epoch train cost = %f' % this_epoch_train_cost
        print 'this epoch valid cost = %f' % this_epoch_valid_cost
        print 'this epoch learning rate = %f' % learning_rate_fn(epoch)
        print 'decay = %f' % decay

        weights = [i.get_value(borrow=True) for i in params]
        np.save(os.path.join(base_path,'weights_'+base_name+'_epoch%04d'%epoch), weights)
    
    f.close()
    g.close()
    end_time = time.clock()
    print ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate_lenet5(base_path="delme")
