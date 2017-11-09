__author__ = 'Christoph Heindl'
__copyright__ = 'Copyright 2017, Profactor GmbH'
__license__ = 'BSD'

import numpy as np

def sensor_unproject(xy, d, Kinv):
    """Reproject pixels to Cartesian space using depth information."""
    xyo = np.ones((xy.shape[0], 3))
    xyo[:,:-1] = xy
    return (Kinv.dot(xyo.T)*d[None, :]).T

def create_batches(array, batch_size, pad=False):
    """Split dataset into batches of equal size."""
    ix = np.arange(batch_size, array.shape[0], batch_size)
    batches = np.split(array, ix, 0)
    if len(batches[-1]) != batch_size:
        b = np.zeros(batches[0].shape)
        b[:batches[-1].shape[0]] = batches[-1]
        batches[-1] = b
    return batches

def mask_outliers(array, spread=1.5):
    """Identifiy outliers using interquatile range statistics."""
    a = array.ravel()
    q = np.percentile(a.ravel(), [25,50,75])
    qrange = q[-1] - q[0]            
    outliers = (a > (q[1] + spread*qrange)) | (a < (q[1] - spread*qrange))
    return outliers.reshape(array.shape)