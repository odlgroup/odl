# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Utilities for computing statistics on images."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('mse', 'psnr', 'estimate_noise')


def mse(true, noisy):
    """Return the Mean Squared Errror.

    Parameters
    ----------
    true : array-like
    noisy : array-like

    Returns
    -------
    mse : float

    Examples
    --------
    >>> true = [1, 1, 1, 1, 1]
    >>> noisy = [1, 1, 1, 1, 2]
    >>> result = mse(true, noisy)
    >>> print('{:.3f}'.format(result))
    0.200
    """
    true = np.asarray(true)
    noisy = np.asarray(noisy)
    return np.mean((true - noisy) ** 2)


def psnr(true, noisy):
    """Return the Peak signal-to-noise ratio.

    Parameters
    ----------
    true : array-like
    noisy : array-like

    Returns
    -------
    psnr : float

    Examples
    --------
    >>> true = [1, 1, 1, 1, 1]
    >>> noisy = [1, 1, 1, 1, 2]
    >>> result = psnr(true, noisy)
    >>> print('{:.3f}'.format(result))
    6.990

    If true == noisy, the result is infinite

    >>> psnr([1, 1], [1, 1])
    inf

    If `true == 0` but `noisy != 0`, the result is negative infinity

    >>> psnr(0, 1)
    -inf
    """
    true = np.asarray(true)
    noisy = np.asarray(noisy)

    mse_result = mse(true, noisy)
    max_true = np.max(np.abs(true))

    if mse_result == 0:
        return np.inf
    elif max_true == 0:
        return -np.inf
    else:
        return 20 * np.log10(max_true) - 10 * np.log10(mse_result)


def estimate_noise(img):
    """Estimate noise level in image.
    
    Parameters
    ----------
    img : array-lik
    
    Returns
    -------
    noise : float
        estimate of standard deviation in image
        
    Examples
    --------
    Create image with noise 1.0, verify result
    
    >>> img = np.random.randn(10, 10)
    >>> noise = estimate_noise(img)  # should be about 1
    
    See Also
    --------
    Reference: J. Immerkaer, “Fast Noise Variance Estimation”
    """
    import scipy.signal
    import functools
    img = np.asarray(img, dtype='float')
    
    M = functools.reduce(np.add.outer, [[-1, 2, -1]] * img.ndim)
    
    convolved = scipy.signal.convolve(img, M, 'valid')
    conv_var = np.sum(np.square(convolved))
    
    sigma = np.sqrt(conv_var / (np.sum(M**2) * np.prod(convolved.shape)))
    
    return sigma


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
