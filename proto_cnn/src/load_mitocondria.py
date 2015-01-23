# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 18:17:44 2014

@author: vivianapetrescu
"""
import misc
import patches
import numpy as np
import theano
import theano.tensor as T

def load_mitocondria(frame = None):
     print "Loading data for frame ... ", frame

     # Load datasets
     path_to_data = '../data/'
     if frame == None:
     	 train_set_x = np.load(path_to_data + 'train_set_x_1000000_51.npy')
     	 train_set_y = np.load(path_to_data + 'train_set_y_1000000_51.npy')
     	 valid_set_x = np.load(path_to_data + 'valid_set_x_200000_51.npy')
     	 valid_set_y = np.load(path_to_data +'valid_set_y_200000_51.npy')
     	 test_set_x = np.load(path_to_data +'test_set_x_fr1_51.npy')
     	 test_set_y = np.load(path_to_data +'test_set_y_fr1_51.npy')
     else:
     	 train_set_x = np.load(path_to_data + 'train_set_x_100000_51.npy')
     	 train_set_y = np.load(path_to_data + 'train_set_y_100000_51.npy')
     	 valid_set_x = np.load(path_to_data + 'valid_set_x_20000_51.npy')
     	 valid_set_y = np.load(path_to_data +'valid_set_y_20000_51.npy')
    	 Vtest = (misc.tiffread('../data/Volume_test.tif')/255.0)
    	 Gtest = (misc.tiffread('../data/Ground_truth_test.tif'))
    	 Gtest[Gtest==255] = 1
         areasize = 51
         gap = (areasize -1)/2

        # Extract all 51x51 patches from testing data
       	 centers = patches.grid_patches_centers(Vtest[frame,:,:].shape, (areasize,areasize))
         test_set_x = patches.get_many_patches(Vtest[frame,:,:], (areasize,areasize), centers, flat=True)
         test_set_y = Gtest[frame][centers[:,0], centers[:,1]]
         print test_set_x.shape
         print test_set_y.shape
     
     def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
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

     train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))
     valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
     test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
     rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
     return rval
