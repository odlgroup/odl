# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# The images are (c) by Matthias Joachim Ehrhardt, University of Cambridge.
#
# The images are licensed under a
# Creative Commons Attribution 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see <http://creativecommons.org/licenses/by/4.0/>.


"""Images provided by the University of Cambridge."""

import numpy as np
import skimage.transform
from odl.contrib.datasets.util import get_data

__all__ = ('brain_phantom', 'resolution_phantom', 'building', 'rings',
           'blurring_kernel')


DATA_SUBSET = 'images_cambridge'
URL_CAM = 'https://raw.github.com/mehrhardt/spdhg/master/data/'


def convert(image, shape, gray=False, dtype='float64', normalize='max'):
    """Convert image to standardized format.

    Several properties of the input image may be changed including the shape,
    data type and maximal value of the image. In addition, this function may
    convert the image into an ODL object and/or a gray scale image.
    """

    image = image.astype(dtype)

    if gray:
        image[..., 0] *= 0.2126
        image[..., 1] *= 0.7152
        image[..., 2] *= 0.0722
        image = np.sum(image, axis=2)

    if shape is not None:
        image = skimage.transform.resize(image, shape, mode='constant')
        image = image.astype(dtype)

    if normalize == 'max':
        image /= image.max()
    elif normalize == 'sum':
        image /= image.sum()
    else:
        assert False

    return image


def brain_phantom(shape=None):
    """Brain phantom for FDG PET simulations.

    Returns
    -------
    An image with the following properties:
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

    return convert(im, shape)


def resolution_phantom(shape=None):
    """Resolution phantom for tomographic simulations.

    Returns
    -------
    An image with the following properties:
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

    return convert(im, shape)


def building(shape=None, gray=False):
    """Photo of the Centre for Mathematical Sciences in Cambridge.

    Returns
    -------
    An image with the following properties:
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

    return convert(im, shape, gray=gray)


def rings(shape=None, gray=False):
    """Photo of married couple holding hands.

    Returns
    -------
    An image with the following properties:
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

    return convert(im, shape, gray=gray)


def blurring_kernel(shape=None):
    """Blurring kernel for convolution simulations.

    The kernel is scaled to sum to one.

    Returns
    -------
    An image with the following properties:
        image type: gray scales
        size: [100, 100] (if not specified by `size`)
        scale: [0, 1]
        type: float64

    """
    # TODO: Store data in some ODL controlled url
    name = 'motionblur.mat'
    url = URL_CAM + name
    dct = get_data(name, subset=DATA_SUBSET, url=url)

    return convert(255 - dct['im'], shape, normalize='sum')


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
