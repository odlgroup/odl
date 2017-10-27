# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Images provided by the University of Cambridge."""

import numpy as np
import odl
from odl.contrib.datasets.util import get_data
from scipy.misc import imresize

__all__ = ('brain_phantom', 'resolution_phantom', 'building', 'rings',
           'blurring_kernel')


DATA_SUBSET = 'images_cambridge'
URL_CAM = 'http://store.maths.cam.ac.uk/DAMTP/me404/data_sets/'


def convert(image, shape, to_odl, gray=None, dtype='float64', normalize='max'):
    """Convert image an ODL object in standardized format.

    In addition several properties of the input image may be changed including
    the shape, data type and maximal value of the image. In addition, this
    function may convert the image into an ODL object and/or a gray scale
    image.
    """

    image.astype(dtype)

    if gray is not None and gray:
        image = np.mean(image, 2)

    if shape is not None:
        image = imresize(image, shape)

    if normalize == 'max':
        image = image / image.max()

    elif normalize == 'sum':
        image = image / image.sum()

    if to_odl:
        shape = image.shape

        if len(shape) == 2:
            space = odl.uniform_discr([0, 0], shape, shape)
        elif len(shape) == 3:
            d = shape[2]
            shape = shape[:2]
            image = np.transpose(image, (2, 0, 1))
            space = odl.uniform_discr([0, 0], shape, shape) ** d

        image = space.element(image)

    return image


def brain_phantom(shape=None, to_odl=True):
    """Brain phantom for FDG PET simulations.

    Output:
        image type: gray scales
        shape: [1024, 1024] (if not specified by `size`)
        scale: [0, 1]
        type: float64
    """
    # TODO: Store data in some ODL controlled url
    name = 'PET_phantom.mat'
    url = URL_CAM + name
    dct = get_data(name, subset=DATA_SUBSET, url=url)
    im = np.rot90(dct['im'], k=3)

    return convert(im, shape, to_odl)


def resolution_phantom(shape=None, to_odl=True):
    """Resolution phantom for tomographic simulations.

    Output:
        image type: gray scales
        shape: [1024, 1024] (if not specified by `size`)
        scale: [0, 1]
        type: float64
    """
    # TODO: Store data in some ODL controlled url
    # TODO: This can be also done with ODL's ellipse_phantom
    name = 'phantom_resolution.mat'
    url = URL_CAM + name
    dct = get_data(name, subset=DATA_SUBSET, url=url)
    im = np.rot90(dct['im'], k=3)

    return convert(im, shape, to_odl)


def building(shape=None, gray=False, to_odl=True):
    """Photo of the Centre for Mathematical Sciences in Cambridge.

    Output:
        image type: color (or gray scales if `gray=True`)
        size: [442, 331] (if not specified by `size`)
        scale: [0, 1]
        type: float64
    """
    # TODO: Store data in some ODL controlled url
    name = 'cms.mat'
    url = URL_CAM + name
    dct = get_data(name, subset=DATA_SUBSET, url=url)
    im = np.rot90(dct['im'], k=3)

    return convert(im, shape, to_odl, gray=gray)


def rings(shape=None, gray=False, to_odl=True):
    """Photo of married couple holding hands.

    Output:
        image type: color (or gray scales if `gray=True`)
        size: [3264, 2448] (if not specified by `size`)
        scale: [0, 1]
        type: float64
    """
    # TODO: Store data in some ODL controlled url
    name = 'rings.mat'
    url = URL_CAM + name
    dct = get_data(name, subset=DATA_SUBSET, url=url)
    im = np.rot90(dct['im'], k=2)

    return convert(im, shape, to_odl, gray=gray)


def blurring_kernel(shape=None, to_odl=True):
    """Blurring kernel for convolution simulations.

    The kernel is scaled to sum to one.

    Output:
        image type: gray scales
        size: [100, 100] (if not specified by `size`)
        scale: [0, 1]
        type: float64

    """
    # TODO: Store data in some ODL controlled url
    name = 'motionblur.mat'
    url = URL_CAM + name
    dct = get_data(name, subset=DATA_SUBSET, url=url)

    return convert(255 - dct['im'], shape, to_odl, normalize='sum')


if __name__ == '__main__':
    # Run doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
