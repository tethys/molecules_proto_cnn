# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:39:22 2014

@author: vpetresc
"""

import numpy as np
from src.tools import misc
from src.tools import patches


from src.core.test_voc_cnn import CNNTestVOC

class CNNTestVOCMitocondria(CNNTestVOC):
    """ Loads the mitocondria data"""
    def __init__(self, protofile, cached_weights_file, frame=None):
        self.frame = frame
        super(CNNTestVOCMitocondria, self).__init__(protofile, cached_weights_file)

    def load_samples(self):
        path_to_data = '../data/'
        if self.frame != None:
            Volume_test = (misc.tiffread(path_to_data + 'Volume_test.tif') / 255.0)
            GT_test = (misc.tiffread(path_to_data + 'Ground_truth_test.tif'))
            GT_test[Gtest == 255] = 1
            areasize = 51

            # Extract all 51x51 patches from testing data
            centers = patches.grid_patches_centers(Volume_test[self.frame, :, :].shape, (areasize, areasize))
            test_set_x = patches.get_many_patches(Volume_test[self.frame, :, :],
                                                  (areasize, areasize),
                                                  centers, flat=True)
            test_set_y = GT_test[self.frame][centers[:, 0], centers[:, 1]]
            print 'Test set shape ', test_set_x.shape
        else:
            test_set_x = np.load(path_to_data + 'test_set_x_fr1_51.npy')
            test_set_y = np.load(path_to_data + 'test_set_y_fr1_51.npy')
            test_set_x, test_set_y = self.shared_dataset((test_set_x, test_set_y))
        return test_set_x, test_set_y
