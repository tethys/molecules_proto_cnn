# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 12:14:03 2014

@author: vpetrescu
"""
import os

def main():
   # loop through the  rank of the first layer
   for r in xrange(14,15,2):
        test_folder = './src/experiments/mnist/test_l1_{0}_l2_50'.format(r)
        if not os.path.exists(test_folder):
            print "Folder does not exist ", test_folder
            sys.exit(1)

        # run retrain and test code
        separable_prototxtfile = test_folder + '/cnn_model_separable.prototxt'
        separable_original_weights = test_folder + '/cnn_separable_model_original.npy'
        if not os.path.exists(separable_prototxtfile):
            print 'Separable prototxtfile does not exist ', separable_prototxtfile
            sys.exit(1)
        if not os.path.exists(separable_original_weights):
            print 'Separable weights file does not exist'
            sys.exit(1)
        test_command = "python ./scripts/mnist_train_test.py"
        test_command+= " -p " + separable_prototxtfile
        test_command+= " -w " + separable_original_weights
        test_command+= " -r=3"
        print "Test command is ", test_command
        os.system(test_command)

if __name__ == main():
    main()
