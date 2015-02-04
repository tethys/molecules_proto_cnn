# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:39:22 2014

@author: vpetresc
"""

from test_cnn import CNNTest

class CNNTestTP(CNNTest):
    """The class implements the methods for computing the batch
        error using the error rate (TP+TN)/(all samples)

    Args:

    Returns:

    """

    def compute_batch_error(self, batch_result_dict):
        """computes the error rate as the number of correctly
        classified samples

        Args:
          batch_result_dict: 

        Returns:

        """
        TP = batch_result_dict['TP']
        FP = batch_result_dict['FP']
        FN = batch_result_dict['FN']
        TN = batch_result_dict['TN']
        return (TP + TN)/float(TP + TN+ FP + FN)

    def compute_all_samples_error(self, all_samples_result):
        """Computes the all samples error by aggregating the
            results from an array of dictionary results

        Args:
          all_samples_result: 

        Returns:

        """
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for result in all_samples_result:
            TP += result['TP']
            FP += result['FP']
            FN += result['FN']
            TN += result['TN']
        return (TP+TN)/float(TN+ TP + FP + FN)
