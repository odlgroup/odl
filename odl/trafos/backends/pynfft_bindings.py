# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Bindings to the ``pynfft`` back-end for non-uniform Fourier transforms.

The `pynfft <https://pythonhosted.org/pyNFFT/>`_  package is a Python
wrapper around the well-known `NFFT <https://www-user.tu-chemnitz.de/~potts/nfft/>`_
library for non-uniform fast Fourier transforms.
"""
try:
    import pynfft
    PYNFFT_AVAILABLE = True
except ImportError:
    PYNFFT_AVAILABLE = False

__all__ = ('PYNFFT_AVAILABLE', )
