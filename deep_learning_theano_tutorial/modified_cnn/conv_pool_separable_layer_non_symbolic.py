# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:18:23 2014

@author: vivianapetrescu
"""

class LeNetSeparableConvPoolLayerNonSymbolic:
    def __init__(self, rng):
        self.rng = rng
    def run_batch(self, batch = layer_input, 
                  image_shape,
                  filter_shape=(clayer_params.num_filters, nbr_feature_maps, 
                                                     clayer_params.filter_w, clayer_params.filter_w),
                                       poolsize=(self.poolsize, self.poolsize),
                                        W = cached_weights[iter + 1], 
                                        b = theano.shared(cached_weights[iter])
    pass