# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Figures of Merit (FOMs) for measuring image quality without a reference."""

from __future__ import division
import numpy as np

__all__ = ('estimate_noise_std',)


def estimate_noise_std(img):
    """Estimate standard deviation of noise in ``img``.

    The algorithm, given in [Immerkaer1996], estimates the noise in an image
    by

    Parameters
    ----------
    img : array-like

    Returns
    -------
    noise : float

    Examples
    --------
    Create image with noise 1.0, verify result

    >>> img = np.random.randn(10, 10)
    >>> result = estimate_noise_std(img)  # should be about 1

    Also works with higher dimensional arrays

    >>> img = np.random.randn(3, 3, 3)
    >>> result = estimate_noise_std(img)

    References
    ----------
    [Immerkaer1996] Immerkaer, J. *Fast Noise Variance Estimation*.
    Computer Vision and Image Understanding, 1996.
    """
    import scipy.signal
    import functools
    img = np.asarray(img, dtype='float')

    M = functools.reduce(np.add.outer, [[-1, 2, -1]] * img.ndim)

    convolved = scipy.signal.fftconvolve(img, M, mode='valid')
    conv_var = np.sum(convolved ** 2)

    scale = np.sum(np.square(M)) * convolved.size
    sigma = np.sqrt(conv_var / scale)

    return sigma


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
