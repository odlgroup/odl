# Copyright 2014, 2015 The ODL development group
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
import odl
from odl.space.base_ntuples import _TYPE_MAP_R2C, _TYPE_MAP_C2R
from odl.util.testutils import skip_if_no_cuda
from odl.util.utility import (
    is_scalar_dtype, is_real_dtype, is_real_floating_dtype,
    is_complex_floating_dtype, complex_space, real_space)


real_float_dtypes = np.sctypes['float']
dtype_ids = [' {} '.format(dt) for dt in real_float_dtypes]


@pytest.fixture(scope="module", ids=dtype_ids, params=real_float_dtypes)
def real_float_dtype(request):
    return np.dtype(request.param)


complex_float_dtypes = np.sctypes['complex']
dtype_ids = [' {} '.format(dt) for dt in complex_float_dtypes]


@pytest.fixture(scope="module", ids=dtype_ids, params=complex_float_dtypes)
def complex_float_dtype(request):
    return np.dtype(request.param)


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


# ---- Space conversion ---- #


# Testing on discretizations since they also cover FunctionSpace and FnBase

def test_complex_space_discr(real_float_dtype):

    discr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                              dtype=real_float_dtype)
    true_cdiscr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                                    dtype=_TYPE_MAP_R2C[real_float_dtype])

    cdiscr = complex_space(discr)
    assert cdiscr == true_cdiscr


def test_complex_space_identity_on_complex(complex_float_dtype):

    discr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                              dtype=complex_float_dtype)
    assert complex_space(discr) == discr


def test_real_space_discr(complex_float_dtype):

    discr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                              dtype=complex_float_dtype)
    true_rdiscr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                                    dtype=_TYPE_MAP_C2R[complex_float_dtype])

    rdiscr = real_space(discr)
    assert rdiscr == true_rdiscr


def test_real_space_identity_on_real(real_float_dtype):

    discr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                              dtype=real_float_dtype)
    assert real_space(discr) == discr


def test_real_complex_space_mutual_inverse(real_float_dtype):

    if real_float_dtype == np.dtype('float16'):
        return

    discr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                              dtype=real_float_dtype)

    cdiscr = odl.uniform_discr([0, 0], [1, 2], (10, 20),
                               dtype=_TYPE_MAP_R2C[real_float_dtype])

    assert real_space(complex_space(discr)) == discr
    assert complex_space(real_space(cdiscr)) == cdiscr


@skip_if_no_cuda
def test_complex_space_discr_cuda(real_float_dtype):

    if real_float_dtype not in odl.space.cu_ntuples.CUDA_DTYPES:
        return

    discr = odl.uniform_discr([0, 0], [1, 2], (10, 20), impl='cuda',
                              dtype=real_float_dtype)

    with pytest.raises(TypeError):
        complex_space(discr)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
