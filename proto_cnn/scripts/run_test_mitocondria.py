
from convolutional_neural_network_separable import ConvolutionalNeuralNetworkSeparableTest;

from cnn_separable_non_symbolic import ConvolutionalNeuralNetworkNonSymbolic;

prototxt_file = './experiments_mitocondria/test_l1_f6_w10_l2_f6_20_l3_f6_w50_100/cnn_big_model_original.prototxt'
cached_weights_file = './experiments_mitocondria/test_l1_f6_w10_l2_f6_20_l3_f6_w50_100/big_model.npy'
for frame in xrange(318):
	cnn = ConvolutionalNeuralNetworkNonSymbolic(prototxt_file, cached_weights_file)
        cnn.compute_test_error()

	
