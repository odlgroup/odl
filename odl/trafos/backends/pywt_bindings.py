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
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


__all__ = ('PAD_MODES_ODL2PYWT', 'PYWT_SUPPORTED_MODES', 'PYWT_AVAILABLE',
           'pywt_wavelet', 'pywt_pad_mode')


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
    """Convert ODL-style padding mode to pywt-style padding mode."""
    pad_mode = str(pad_mode).lower()
    if pad_mode == 'constant' and pad_const != 0.0:
        raise ValueError('constant padding with constant != 0 not supported '
                         'for `pywt` back-end')
    try:
        return PAD_MODES_ODL2PYWT[pad_mode]
    except KeyError:
        raise ValueError("`pad_mode` '{}' not understood".format(pad_mode))


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests(skip_if=not PYWT_AVAILABLE)
