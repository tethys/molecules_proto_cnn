# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:39:22 2014

@author: vpetresc
"""

from test_cnn import CNNTest

class CNNTestVOC(CNNTest):
    """Computes the VOC error"""

    def compute_batch_error(self, batch_result_dict):
        """Computes the VOC error

        Args:
          batch_result_dict: 

        Returns:

        """
        TP = batch_result_dict['TP']
        FP = batch_result_dict['FP']
        FN = batch_result_dict['FN']
        return TP/float(TP + FP + FN)

    def compute_all_samples_error(self, all_samples_result):
        """Computes the error as VOC across al samples

        Args:
          all_samples_result: 

        Returns:

        """
        TP = 0
        FP = 0
        FN = 0
        for result in all_samples_result:
            TP += result['TP']
            FP += result['FP']
            FN += result['FN']
        return TP/float(TP + FP + FN)
