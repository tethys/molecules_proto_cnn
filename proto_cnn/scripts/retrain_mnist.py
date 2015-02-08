# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 12:14:03 2014

@author: vpetrescu
"""
import os
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format

def main():
   # loop through the  rank of the first layer
   for r in xrange(14,20,2):
        test_folder = './experiments_cnn_paper/test_f6_k9_f12_k9_r{0}'.format(r)
        if os.path.exists(test_folder) == False:
            print "Folder does not exist ", test_folder
            sys.exit(1)

        """Load settings"""
        settings = pb_cnn.CNNSettings();        
        try:
           f = open('./experiments/mnist/cnn_separable_test.prototxt', "r")
           data = f.read()
           print 'Protofile content: %s\n', data
           text_format.Merge(data, settings)
           f.close()
        except IOError:
           print "Could not open file ./experiments/cnn_test.prototxt"
        settings.conv_layer[0].rank = r

        command = 'cp ./experiments_cnn_paper/paper_cvlab_50batches.npy {0}'.format(test_folder)
        os.system(command)    

        # run command generate separable filters
        command = "python generate_separable_weights.py -p "
        command+=  updated_prototxtfile 
        command+= " -w " + test_folder + "/paper_cvlab_50batches.npy"
        command+= " -s " + test_folder + '/cnn_separable_model_original.npy'
        print "Command is ", command
        # run command generate test result
        os.system(command)

        test_command = "python cnn_train_test.py -p "
        test_command+= updated_prototxtfile
        test_command+= " -w " + test_folder + '/cnn_separable_model_original.npy'
        test_command+= " -r=3"
        print "Test command is ", test_command
      #  os.system(test_command)


if __name__ == main():
    main()
