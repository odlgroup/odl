# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:48:55 2016

@author: jonasadler
"""

from odl.discr import uniform_discr_frompartition
import numpy as np
from skimage.transform import radon, iradon

def scikit_radon_forward(x, geometry, range, out):
    image = x.asarray()
    theta = np.asarray(geometry.motion_grid).squeeze()

    midp = x.space.domain.midpoint
    shape = x.space.shape
    extent = x.space.domain.extent()
    assert all(midp == [0, 0])
    assert shape[0] == shape[1]
    assert extent[0] == extent[1]


    scikit_detector_part = odl.uniform_partition(-extent[0] / np.sqrt(2),
                                              extent[0] / np.sqrt(2),
                                              int(np.ceil(shape[0] * np.sqrt(2))))

    scikit_range_part = geometry.motion_partition.insert(1,
                                                         scikit_detector_part)

    scikit_range = uniform_discr_frompartition(scikit_range_part,
                                               interp=range.interp,
                                               dtype=range.dtype)
    sinogram = scikit_range.element(radon(image, theta=theta).T)
    out.sampling(sinogram.interpolation)
