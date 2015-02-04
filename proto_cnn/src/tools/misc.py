# -*- coding: utf-8 -*-

"""
misc
====

This module implements functions which do not have another home.
"""

__author__ = "Pablo Márquez Neila"

import numpy as np
from matplotlib import pyplot
from scipy import ndimage

def grouper(n, iterable):
    return (iterable[i:i+n] for i in xrange(0,len(iterable),n))

def unique_rows(a):
    """Remove duplicated rows in a 2D array."""
    a = np.asarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def sort_nicely( l ): 
    """ Sort the given list in the way that humans expect.""" 
    import re
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key=alphanum_key) 

def matarray(*args, **kwargs):
    """
    Construct a 2D array using a Matlab-like notation.
    You can specify the separator element using the keyword
    argument 'sep'. By default, it is 'None'. This is
    useful when creating block matrices.
    
    When the resulting matrix has only one element, matarray
    will return the element.
    
    Examples:
    >>> matarray(1,2,None,3,4)
    array([[1, 2],
           [3, 4]])
    >>> matarray(1, 2, '', 3, 4, sep='')
    array([[1, 2],
           [3, 4]])
    >>> R = np.ones((3,3))
    >>> t = np.zeros((3,1))
    >>> matarray(R, t, None, 0, 0, 0, 1)
    array([[ 1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])
    >>> matarray([3])
    3
    """
    sep = None
    arr = list(args)
    if len(arr) == 1:
        arr = arr[0]
    
    if not isinstance(arr, list):
        return arr
    
    if kwargs.has_key('sep'):
        sep = kwargs['sep']
    
    res = []
    aux = []
    for e in arr:
        if e == sep:
            # New row.
            if len(aux) > 0:
                res.append(np.hstack(aux))
            aux = []
        elif hasattr(e, '__iter__'):
            # Sub-matrix.
            submat = matarray(e, sep=sep)
            if submat is not None:
                aux.append(submat)
        else:
            aux.append(e)
    
    if len(aux) > 0:
        res.append(np.hstack(aux))
    if len(res) > 0:
        res = np.vstack(res)
        # If res has only one element, return the element.
        if res.size == 1:
            return res[0,0]
        return res
    
    return None

class redict(dict):
    """Dictionary indexed by regular expressions."""
    import re
    
    def __repr__(self):
        return "redict(%s)" % dict.__repr__(self)
    
    def __getitem__(self, skey):
        res = {}
        for key, value in self.iteritems():
            if self.re.search(skey, key):
                res[key] = value
        if len(res) == 0:
            raise KeyError, skey
        return res

def mklist(lst, dim):
    """
    If *lst* is a scalar value or a sequence with one element, returns a list
    of that element repeated *dim* times.
    
    If *lst* is a sequence of length *dim*, returns *lst*.
    
    Otherwise, it raises a ValueError exception.
    
    Examples:
    >>> mklist(1, 4)
    [1, 1, 1, 1]
    >>> mklist([2], 4)
    [2, 2, 2, 2]
    >>> mklist([1, 2, 3, 4], 4)
    [1, 2, 3, 4]
    >>> mklist([1, 2, 3], 4)
    ValueError: lst has an invalid length
    """
    try:
        d = len(lst)
        if d == 1:
            lst = lst * dim
        elif d != dim:
            raise ValueError, "lst has an invalid length"
    except TypeError:
        lst = [lst] * dim
    return lst

def plot_grid(elements, plotfun, width=None):
    """
    Plot a set of elements in a grid of subplots.
    
    Parameters
    ----------
    elements : list
        List of any type.
    plotfun : function
        The function *plotfun* should accept two parameters:
        plotfun(axes, element). *axes* is a instance of matplotlib
        AxesSubplot, and *element* is an element of the list *elements*.
        *plotfun* should plot the element in the given AxesSubplot.
    width : int
        If given, it determines the width of the plot grid. If not given,
        the width of the grid is the square root of the length of *elements*.
    """
    # Determine the size of the plot.
    if width is None:
        width = np.int_(np.ceil(np.sqrt(len(elements))))
    height = np.int_(np.ceil(len(elements) / float(width)))
    
    # Get a new figure.
    fig = pyplot.gcf()
    fig.clf()
    for idx, elem in enumerate(elements):
        ax = fig.add_subplot(height, width, idx+1)
        plotfun(ax, elem)
    fig.show()

def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

def srgb2ciexyz(img):
    if img.dtype == np.uint8:
        img = img/255.0
    
    g = np.zeros_like(img)
    g[img<=0.04045] = img[img<=0.04045]/12.92
    g[img>0.04045] = ((img[img>0.04045]+0.055)/1.055)**2.4
    g = np.reshape(g, (-1, 3)).T
    
    mat = np.array([[0.4124, 0.3576, 0.1805],
                    [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]])
    res = np.dot(mat, g).T
    return np.reshape(res, img.shape)

def srgb2cielab(img):
    img = srgb2ciexyz(img)
    ciexyz_white = np.array([0.9505, 1.0, 1.089])
    img = img/ciexyz_white
    f = np.zeros_like(img)
    
    c = (6/29.)**3
    f[img > c] = img[img > c]**(1.0/3.0)
    f[img < c] = img[img < c] * (29.0/6.0)**2 / 3 + 4.0/29.0
    res = np.zeros_like(img)
    res[...,0] = 116*f[...,1] - 16
    res[...,1] = 500*(f[...,0] - f[...,1])
    res[...,2] = 200*(f[...,1] - f[...,2])
    return res

def colorzoom(img, zoom):
    
    img1 = ndimage.zoom(img[...,0], zoom)
    res = np.empty(img1.shape + (img.shape[-1],), img1.dtype)
    res[...,0] = img1
    for i in xrange(1,img.shape[-1]):
        res[...,i] = ndimage.zoom(img[...,i], zoom)
    return res

def tiffread(fname, strip=False):
    """Read a multi-page TIFF image to a three-dimensional array."""
    import Image
    import itertools
    img = Image.open(fname)
    
    res = []
    offsets = []
    bbox = []
    frame = 0
    try:
        for frame in itertools.count():
            img.seek(frame)
            aux = np.asarray(img)
            if aux.ndim == 0:
                if img.mode == 'I;16':
                    aux = np.fromstring(img.tostring(), np.uint16)
                    aux = np.reshape(aux, img.size[::-1])
                elif img.mode == 'I;16S':
                    aux = np.fromstring(img.tostring(), np.int16)
                    aux = np.reshape(aux, img.size[::-1])
                else:
                    raise ValueError, "unknown pixel mode"
            # Strip the image if required.
            if strip:
                bbox.append(img.getbbox())
            res.append(aux)
    except EOFError:
        pass
    
    res = np.asarray(res)
    if strip:
        bbox = np.asarray(bbox)[:,[1,0,3,2]]
        upper = bbox[:,:2].max(0)
        lower = bbox[:,2:].min(0)
        slices = (slice(None), slice(upper[0],lower[0]), slice(upper[1],lower[1]))
        res = res[slices]
        return res, slices
    return res

def npz2dict(npz):
    """Convert a npz structure in a Python dictionary."""
    res = {}
    for i in npz.files:
        res[i] = npz[i]
    return res

def loadz(filename, destdict=None):
    """
    Load a .npz file and insert its contents into a namespace.
    
    If destdict is not given, the contents will be inserted
    in the global namespace of the calling frame.
    
    Example:
    >>> import numpy as np
    >>> A = np.array([1,2,3])
    >>> b = np.array([[1,2],[3,4]])
    >>> np.savez("test", A=A, b=b)
    >>> del A, b
    >>> loadz("test.npz")
    >>> A
    array([1, 2, 3])
    >>> b
    array([[1, 2],
           [3, 4]])
    """
    if destdict is None:
        import sys
        destdict = sys._getframe().f_back.f_globals
    
    npz = np.load(filename)
    dct = npz2dict(npz)
    destdict.update(dct)

def detect_peaks(img, minimum_distance=None, window_size=3):
    """Detect prominent peaks in an image.
    
    If given, ``minimum_distance`` indicates the minimum distance
    that each detected peak is from its nearest neighbor.
    
    ``window_size`` indicates the size of the window
    for the non-maxima suppression algorithm. The default is (3,3).
    
    From Yanxi Liu, Robert T. Collins, 
    "A Computational Model for Repeated Pattern Perception using
    Frieze and Wallpaper Groups"
    """
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    
    # Look for local maxima.
    localmaxima = ndimage.grey_dilation(img, size=mklist(window_size, img.ndim)) == img
    coords = np.nonzero(localmaxima)
    num_maxima = coords[0].shape[0]
    # Sort the local maxima by its magnitude in the input image.
    order = img[coords].argsort()[::-1]
    coords_sorted = np.array([i[order] for i in coords]).T
    
    # Look for isolated local maxima.
    distances = np.zeros(num_maxima)
    distances[0] = np.inf
    for idx, peak in enumerate(coords_sorted[1:], 1):
        d = cdist([peak], coords_sorted[:idx])
        distances[idx] = d.min(1)
    order = distances.argsort()[::-1]
    coords_sorted = coords_sorted[order]
    
    # Prune close neighbors if required.
    if minimum_distance is not None:
        last_index = np.nonzero(distances[order] < minimum_distance)[0][0]
        return coords_sorted[:last_index]
    
    return coords_sorted

def whiten(data, epsilon=1e-3):
    
    # Subtract the mean of each data point.
    data = data - data.mean(1)[:, np.newaxis]
    
    # PCA.
    cov = np.cov(data.T)
    U, D, V = np.linalg.svd(cov)
    
    data_pca = np.dot(U.T / np.sqrt(D + epsilon)[:, np.newaxis], data.T).T
    data_zca = np.dot(U, data_pca.T).T
    
    return data_zca

def normalize_data(data):
    
    data = data - data.mean(0)
    
    pstd = 3 * data.std()
    data = np.clip(data, -pstd, pstd) / pstd
    
    return (data + 1) * 0.4 + 0.1
