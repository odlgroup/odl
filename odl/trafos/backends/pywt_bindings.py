# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bindings to the PyWavelets backend for wavelet transforms.

`PyWavelets <https://pywavelets.readthedocs.io/>`_ is a Python library
for wavelet transforms in arbitrary dimensions, featuring a large number
of built-in wavelet filters.
"""

from __future__ import print_function, division, absolute_import

import numpy as np

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


__all__ = ('PAD_MODES_ODL2PYWT', 'PYWT_SUPPORTED_MODES', 'PYWT_AVAILABLE',
           'pywt_wavelet', 'pywt_pad_mode', 'precompute_raveled_slices')


# A clear illustration of all of these padding modes is available at:
# https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
PAD_MODES_ODL2PYWT = {'constant': 'zero',
                      'periodic': 'periodic',
                      'symmetric': 'symmetric',
                      'order0': 'constant',
                      'order1': 'smooth',
                      'pywt_periodic': 'periodization',
                      'reflect': 'reflect',
                      'antireflect': 'antireflect',
                      'antisymmetric': 'antisymmetric',
                      }
PYWT_SUPPORTED_MODES = PAD_MODES_ODL2PYWT.values()


def pywt_wavelet(wavelet):
    """Convert ``wavelet`` to a `pywt.Wavelet` instance."""
    if isinstance(wavelet, pywt.Wavelet):
        return wavelet
    else:
        return pywt.Wavelet(wavelet)


def pywt_pad_mode(pad_mode, pad_const=0):
    """Convert ODL-style padding mode to pywt-style padding mode.

    Parameters
    ----------
    pad_mode : str
        The ODL padding mode to use at the boundaries.
    pad_const : float, optional
        Value to use outside the signal boundaries when ``pad_mode`` is
        'constant'. Only a value of 0. is supported by PyWavelets

    Returns
    -------
    pad_mode_pywt : str
        The corresponding name of the requested padding mode in PyWavelets.
        See `signal extension modes`_.

    References
    ----------
    .. _signal extension modes:
       https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html
    """
    pad_mode = str(pad_mode).lower()
    if pad_mode == 'constant' and pad_const != 0.0:
        raise ValueError('constant padding with constant != 0 not supported '
                         'for `pywt` back-end')
    try:
        return PAD_MODES_ODL2PYWT[pad_mode]
    except KeyError:
        raise ValueError("`pad_mode` '{}' not understood".format(pad_mode))


def precompute_raveled_slices(coeff_shapes, axes=None):
    """Return slices and shapes for raveled multilevel wavelet coefficients.

    The output is equivalent to the ``coeff_slices`` output of
    `pywt.ravel_coeffs`, but this function does not require computing a
    wavelet transform first.

    Parameters
    ----------
    coeff_shapes : array-like
        A list of multilevel wavelet coefficient shapes as returned by
        `pywt.wavedecn_shapes`.
    axes : sequence of ints, optional
        Axes over which the DWT that created ``coeffs`` was performed. The
        default value of None corresponds to all axes.

    Returns
    -------
    coeff_slices : list
        List of slices corresponding to each coefficient. As a 2D example,
        ``coeff_arr[coeff_slices[1]['dd']]`` would extract the first level
        detail coefficients from ``coeff_arr``.

    Examples
    --------
    >>> import pywt
    >>> data_shape = (64, 64)
    >>> coeff_shapes = pywt.wavedecn_shapes(data_shape, wavelet='db2', level=3,
    ...                                     mode='periodization')
    >>> coeff_slices = precompute_raveled_slices(coeff_shapes)
    >>> print(coeff_slices[0])  # approximation coefficients
    slice(None, 64, None)
    >>> d1_coeffs = coeff_slices[-1]  # first level detail coefficients
    >>> (d1_coeffs['ad'], d1_coeffs['da'], d1_coeffs['dd'])
    (slice(1024, 2048, None), slice(2048, 3072, None), slice(3072, 4096, None))
    """
    # initialize with the approximation coefficients.
    a_shape = coeff_shapes[0]
    a_size = np.prod(a_shape)

    if len(coeff_shapes) == 1:
        # only a single approximation coefficient array was found
        return [slice(a_size), ]

    a_slice = slice(a_size)

    # initialize list of coefficient slices
    coeff_slices = []
    coeff_slices.append(a_slice)

    # loop over the detail cofficients, embedding them in coeff_arr
    details_list = coeff_shapes[1:]
    offset = a_size
    for shape_dict in details_list:
        # new dictionaries for detail coefficient slices and shapes
        coeff_slices.append({})
        keys = sorted(shape_dict.keys())
        for key in keys:
            shape = shape_dict[key]
            size = np.prod(shape)
            sl = slice(offset, offset + size)
            offset += size
            coeff_slices[-1][key] = sl
    return coeff_slices


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYWT_AVAILABLE)
