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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

# External
import pytest
import numpy as np

# Internal
from odl.util.utility import (
    is_scalar_dtype, is_real_dtype, is_real_floating_dtype,
    is_complex_floating_dtype)


real_float_dtypes = np.sctypes['float']
complex_float_dtypes = np.sctypes['complex']
nonfloat_scalar_dtypes = np.sctypes['uint'] + np.sctypes['int']
scalar_dtypes = (real_float_dtypes + complex_float_dtypes +
                 nonfloat_scalar_dtypes)
real_dtypes = real_float_dtypes + nonfloat_scalar_dtypes
# Need to make concrete instances here (with string lengths)
nonscalar_dtypes = [np.dtype('S1'), np.dtype('<U2'), np.dtype(object),
                    np.dtype(bool), np.void]


# ---- Data type helpers ---- #


def test_is_scalar_dtype():
    for dtype in scalar_dtypes:
        assert is_scalar_dtype(dtype)


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
