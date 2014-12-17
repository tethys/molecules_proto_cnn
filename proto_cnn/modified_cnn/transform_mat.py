# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 21:55:05 2014

@author: vivianapetrescu
"""

import scipy.io as sio
import numpy as np
#>>> import numpy as np
#>>> vect = np.arange(10)
#>>> vect.shape
#sio.savemat('np_vector.mat', {'vect':vect})
#mat_contents = sio.loadmat('octave_a.mat')

params = np.load('./experiments/setup_theano_tutorial/cnn_model_original2.npy')
sio.savemat('./experiments/setup_theano_tutorial/cnn_model_original2.mat',{'params':params})