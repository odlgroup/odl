# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Figures of Merit (FOMs) for measuring image quality without a reference."""

import odl
import numpy as np

__all__ = ('estimate_noise_std',)


def estimate_noise_std(img, average=True):
    """Estimate standard deviation of noise in ``img``.

    The algorithm, given in [Immerkaer1996], estimates the noise in an image.

    Parameters
    ----------
    img : `odl.Tensor`
        Array to estimate noise in.
    average : bool
        If ``True``, return the mean noise in the image, otherwise give a
        pointwise estimate.

    Returns
    -------
    noise : float

    Examples
    --------
    Create image with noise 1.0, verify result

    >>> space = odl.rn((10, 10))
    >>> img = space.element(np.random.randn(*space.shape))
    >>> result = estimate_noise_std(img)  # should be about 1

    Also works with higher dimensional arrays

    >>> space = odl.rn((3, 3, 3))
    >>> img = space.element(np.random.randn(*space.shape))
    >>> result = estimate_noise_std(img)

    The method can also estimate the noise pointwise (but with high
    uncertainty):

    >>> result = estimate_noise_std(img, average=False)

    References
    ----------
    [Immerkaer1996] Immerkaer, J. *Fast Noise Variance Estimation*.
    Computer Vision and Image Understanding, 1996.
    """
    import scipy.signal
    import functools

    space = img.space
    backend = space.array_backend
    device = space.device
    ns = backend.array_namespace

    img = img.astype('float32').asarray()

    M = backend.array_constructor(
            functools.reduce(np.add.outer, [[-1, 2, -1]] * img.ndim)
          , device=device)

    if average:
        # TODO (Justus) it does not really make sense to use FFT for convolving
        # with a small, fixed kernel
        convolved = scipy.signal.fftconvolve(img, M, mode='valid')
        conv_var = ns.sum(convolved ** 2) / convolved.size
    else:
        convolved = scipy.signal.fftconvolve(img, M, mode='same')
        conv_var = convolved ** 2

    scale = ns.sum(ns.square(M))
    sigma = ns.sqrt(conv_var / scale)

    return sigma if average else space.element(sigma)


if __name__ == '__main__':
    from odl.core.util.testutils import run_doctests
    run_doctests()
