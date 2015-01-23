
"""

"""

from operator import attrgetter, methodcaller
from itertools import repeat, product
from functools import partial

import sys
import copy
import numpy as np

def grid_patches_centers(img_shape, patch_shape, step=1):
    """Return the centers of a grid of patches with given shape"""
    center_slices = tuple(slice(i//2, j - (i-i//2) + 1, step)
        for i,j in zip(patch_shape, img_shape))
    return np.reshape(np.mgrid[center_slices], (len(patch_shape),-1)).T

def get_patch(image, patch_shape, center):
    """Return a single patch with the given shape and center"""
    slices = tuple(slice(i-ps//2, i-ps//2+ps) for i,ps in zip(center, patch_shape))
    return image[slices]

def get_grid_patches(image, patch_shape, step=1, flat=True):
    """Return all the patches in a grid"""
    centers = grid_patches_centers(image.shape, patch_shape, step)
    return get_many_patches(image, patch_shape, centers, flat)

def get_many_patches(image, patch_shape, centers, flat=True):
    """Return the patches at given centers"""
    grid_slices = tuple(slice(-(i//2), i-i//2) for i in patch_shape)
    grid = np.reshape(np.mgrid[grid_slices], (2, -1))
    points = tuple(np.int_(centers.T[:,:,np.newaxis]) + np.int_(grid[:,np.newaxis,:]))
    
    patches = image[points]
    if not flat and image.ndim == 2:
        patches = np.reshape(patches, (-1,) + tuple(patch_shape))
    elif not flat and image.ndim == 3:
        patches = np.reshape(patches, (len(patches),) + tuple(patch_shape) + (-1,))
    else:
        patches = np.reshape(patches, (len(patches),-1))
    return patches
