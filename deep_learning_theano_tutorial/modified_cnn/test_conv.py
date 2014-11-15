# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 22:45:57 2014

@author: vivianapetrescu
"""


import numpy as np
import time

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


#input: (batch size, channels, rows, columns)
#filters: (number of filters, channels, rows, columns)


images = T.tensor4('images')
filters = T.tensor4('filters')

image_shape = (100, 3, 5, 5)
input_images = np.ones(image_shape)
horizontal_filters_shape = (11, 3, 3, 3)
horizontal_filters = np.ones(horizontal_filters_shape)
conv_out = conv.conv2d(input = input_images, filters = horizontal_filters,
                               filter_shape = horizontal_filters_shape, image_shape = image_shape,
                               border_mode = 'valid',subsample=(1,1))
print conv_out.shape.eval()                    
                



