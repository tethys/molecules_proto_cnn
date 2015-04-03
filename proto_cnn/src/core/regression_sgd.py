"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets, and a conjugate gradient optimization method
that is suitable for smaller datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class Regression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W = None, b = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        else:
            self.W = W

        # initialize the baises b as a vector of 0's
        if b is None:
            self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        else:
            self.b = b 

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.dot(input, self.W) + self.b
        self.p_y_given_x = T.nnet.softmax(self.p_y_given_x)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # compute prediction as class whose probability is maximal in
        # symbolic form

        # parameters of the model
        self.params = [self.W, self.b]

    def squared_loss(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return T.sqrt(T.mean((self.p_y_given_x - y)**2))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.p_y_given_x.type))
        # check if y is of the correct datatype
        return T.sqrt(T.mean((self.y_pred - y)**2))
        if y.dtype.startswith('int'):
	    return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def result_count_dictionary(self, y):
	""" Returns a dictionary containing the count
            of the TP, TN, FP, FN counts. They are
            needed for aggregating the results over the whole
            batches.
        """
        result = {}
	result['TP'] = self.true_positives_count(y).eval()
	result['TN'] = self.true_negatives_count(y).eval()
	result['FP'] = self.false_positives_count(y).eval()
	result['FN'] = self.false_negatives_count(y).eval()
        return result

    def true_negatives_count(self, y):
        """ Compute the number of true negatives.
            Raises exception if the target value y is not int.

            :param y: array with class index values
            :type y: tensor int array
            :returns: the true negatives count

            :raises: NotImplementedError, TypeError
        """
        self.valid_target_y(y)
        TP = T.sum(T.and_(T.eq(self.y_pred, 0), T.eq(y, 0)))
	return TP

    def true_positives_count(self, y):
        """ Compute the number of true positives.
            Raises exception if the target value y is not int.

            :param y: array with class index values
            :type y: tensor int array
            :returns: the true positives count

            :raises: NotImplementedError, TypeError
        """
        self.valid_target_y(y)
        TP = T.sum(T.and_(T.eq(self.y_pred, 1), T.eq(y, 1)))
        return TP

    def false_positives_count(self, y):
        """ Compute the number of false positives.
            Raises exception if the target value y is not int.

            :param y: array with class index values
            :type y: tensor int array
            :returns: the false positives count

            :raises: NotImplementedError, TypeError
        """
        self.valid_target_y(y)
        FP = T.sum(T.and_(T.eq(y,0), T.neq(self.y_pred, y)))
        return FP

    def false_negatives_count(self, y):
        """ Compute the number of false negatives.
            Raises exception if the target value y is not int.

            :param y: array with class index values
            :type y: tensor int array
            :returns: the false negatives count

            :raises: NotImplementedError, TypeError
        """
        self.valid_target_y(y)
        FN = T.sum(T.and_(T.eq(y,1), T.neq(self.y_pred, y)))
	return FN

    def false_result_count(self, y):
        """ Compute the number of false negatives.
            Raises exception if the target value y is not int.

            :param y: array with class index values
            :type y: tensor int array
            :returns: the false negatives count

            :raises: NotImplementedError, TypeError
        """
        self.valid_target_y(y)
        F = T.sum(T.neq(self.y_pred, y))
	return F

    def true_result_count(self, y):
	self.valid_target_y(y)
	T = T.sum(T.eq(self.y_pred, y))
	return T

    def valid_target_y(self, y):
        """ Verifies that the target values y are integers
            and that the array has same size as the input one.
            :param y: array of target values
            :type y: array of int
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if not y.dtype.startswith('int'):
            raise NotImplementedError('Type of target value y must be int')

