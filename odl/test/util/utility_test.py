# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import pytest
import numpy as np

from odl.util.utility import (
    is_numeric_dtype, is_real_dtype, is_real_floating_dtype,
    is_complex_floating_dtype)


real_float_dtypes = np.sctypes['float']
complex_float_dtypes = np.sctypes['complex']
nonfloat_numeric_dtypes = np.sctypes['uint'] + np.sctypes['int']
numeric_dtypes = (real_float_dtypes + complex_float_dtypes +
                  nonfloat_numeric_dtypes)
real_dtypes = real_float_dtypes + nonfloat_numeric_dtypes
# Need to make concrete instances here (with string lengths)
nonnumeric_dtypes = [np.dtype('S1'), np.dtype('<U2'), np.dtype(object),
                     np.dtype(bool), np.void]


# ---- Data type helpers ---- #


def test_is_numeric_dtype():
    for dtype in numeric_dtypes:
        assert is_numeric_dtype(dtype)


def test_is_real_dtype():
    for dtype in real_dtypes:
        assert is_real_dtype(dtype)


def test_is_real_floating_dtype():
    for dtype in real_float_dtypes:
        assert is_real_floating_dtype(dtype)


def test_is_complex_floating_dtype():
    for dtype in complex_float_dtypes:
        assert is_complex_floating_dtype(dtype)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
