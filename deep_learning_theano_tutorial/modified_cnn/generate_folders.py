# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 12:14:03 2014

@author: vivianapetrescu
"""
import os
import convolutional_neural_network_settings_pb2 as pb_cnn
from google.protobuf import text_format

def main():
   for r in xrange(38,44,2):
   # for r in xrange(12,45,2):
        # generate folder test_l1_x_l2_y if it does not exist
        test_folder = './experiments/test_l1_20_l2_{0}'.format(r)
        if os.path.exists(test_folder) == False:
            os.makedirs(test_folder)
        # copy cnn_model original
        command = 'cp ./experiments/cnn_model_original.npy {0}'.format(test_folder)
        os.system(command)    
    
        """Load settings"""        
        settings = pb_cnn.CNNSettings();        
        try:        
           f = open('./experiments/cnn_model_original.prototxt', "r")
           data=f.read()
           print 'Protofile content:'
           print data
           text_format.Merge(data, settings);
           f.close();
        except IOError:
           print "Could not open file ./experiments/cnn_model_original.prototxt";
        settings.conv_layer[1].rank = r

        try: 
            updated_prototxtfile = './experiments/test_l1_20_l2_{0}/cnn_model_separable.prototxt'.format(r)
            f = open(updated_prototxtfile, "w")
            f.write(text_format.MessageToString(settings))
            f.close()
        except IOError:
            print "Could not write protofile back"
        # copy prototxt file, update it
            
        command = 'cp ./experiments/cnn_model_original.prototxt {0}'.format(test_folder)
        os.system(command)    
        
        
        # run command generate separable filters
        command = "python generate_separable_weights.py -p "
        command+=  updated_prototxtfile 
        command+= " -w " + test_folder + "/cnn_model_original.npy"
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