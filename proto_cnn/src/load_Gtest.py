import patches
import misc
import numpy as np
if __name__ == '__main__':
    Vtest = (misc.tiffread('../data/Volume_test.tif')/255.0)
    Gtest = (misc.tiffread('../data/Ground_truth_test.tif'))
    Gtest[Gtest==255] = 1

    final_x = []
    final_y = []
    for frame in xrange(318):
	areasize = 51
	gap = (areasize -1)/2
        
        # Extract all 51x51 patches from testing data
        centers = patches.grid_patches_centers(Vtest[frame,:,:].shape, (areasize,areasize))
        test_set_x = patches.get_many_patches(Vtest[frame,:,:], (areasize,areasize), centers, flat=True)
        test_set_y = Gtest[frame][centers[:,0], centers[:,1]]
	print test_set_x.shape
	print test_set_y.shape
	if frame > 0:
	 	final_x = np.concatenate((final_x, test_set_x), axis = 0)
		final_y = np.concatenate((final_y, test_set_y), axis = 0)
	else:
		final_x = test_set_x
		final_y = test_set_y
    numpy.save('test_x_large.npy', final_x)
    numpy.save('test_y_large.npy', final_y)
