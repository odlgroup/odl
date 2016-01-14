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

# External module imports
from itertools import product
from math import pi
import numpy as np
import platform
import pytest

# ODL imports
import odl
from odl.trafos.fourier import (
    reciprocal, _shift_list, dft_preproc_data, dft_postproc_data, dft_call,
    DiscreteFourierTransform)
from odl.util.testutils import all_almost_equal, all_equal, skip_if_no_pyfftw
from odl.util.utility import is_real_dtype


# Pytest fixture


# Simply modify exp_params to modify the fixture
exp_params = [2.0, 1.0, float('inf'), 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]
exp_fixture = pytest.fixture(scope="module", ids=exp_ids, params=exp_params)


@exp_fixture
def exponent(request):
    return request.param


# Simply modify exp_params to modify the fixture
if platform.system() == 'Linux':
    dtype_params = ['float32', 'float64', 'float128',
                    'complex64', 'complex128', 'complex256']
else:
    dtype_params = ['float32', 'float64', 'complex64', 'complex128']
dtype_ids = [' dtype = {} '.format(dt) for dt in dtype_params]
dtype_fixture = pytest.fixture(scope="module", ids=dtype_ids,
                               params=dtype_params)


@dtype_fixture
def dtype(request):
    return request.param


# Simply modify exp_params to modify the fixture
plan_params = ['estimate', 'measure', 'patient', 'exhaustive']
plan_ids = [" planning = '{}' ".format(p) for p in plan_params]
plan_fixture = pytest.fixture(scope="module", ids=plan_ids,
                              params=plan_params)


@plan_fixture
def planning(request):
    return request.param


def test_shift_list():

    length = 3

    # Test single value
    shift = True
    lst = _shift_list(shift, length)

    assert all_equal(lst, [True] * length)

    # Test existing sequence
    shift = (False,) * length
    lst = _shift_list(shift, length)

    assert all_equal(lst, [False] * length)

    # Too long sequence, gets truncated but ok
    shift = (False,) * (length + 1)
    lst = _shift_list(shift, length)

    assert all_equal(lst, [False] * length)

    # Test iterable
    def alternating():
        i = 0
        while 1:
            yield bool(i % 2)
            i += 1

    lst = _shift_list(alternating(), length)

    assert all_equal(lst, [False, True, False])

    # Too short sequence, should raise
    shift = (False,) * (length - 1)
    with pytest.raises(ValueError):
        _shift_list(shift, length)

    # Iterable returning too few entries, should throw
    def alternating_short():
        i = 0
        while i < length - 1:
            yield bool(i % 2)
            i += 1

    with pytest.raises(ValueError):
        _shift_list(alternating_short(), length)


def test_reciprocal_1d_odd():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=11, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Should be symmetric
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)
    assert all_almost_equal(rgrid.center, 0)
    # Zero should be at index n // 2
    assert all_almost_equal(rgrid[n // 2], 0)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # No point should be closer to 0 than half a recip stride
    tol = 0.999 * true_recip_stride / 2
    assert not rgrid.approx_contains(0, tol=tol)


def test_reciprocal_1d_odd_halfcomplex():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=11, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, (n + 1) / 2)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max should be zero
    assert all_almost_equal(rgrid.max_pt, 0)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, (n + 1) / 2)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max point should be half a positive recip stride
    assert all_almost_equal(rgrid.max_pt, -true_recip_stride / 2)


def test_reciprocal_1d_even():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=10, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Should be symmetric
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)
    assert all_almost_equal(rgrid.center, 0)
    # No point should be closer to 0 than half a recip stride
    tol = 0.999 * true_recip_stride / 2
    assert not rgrid.approx_contains(0, tol=tol)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Zero should be at index n // 2
    assert all_almost_equal(rgrid[n // 2], 0)


def test_reciprocal_1d_even_halfcomplex():

    grid = odl.uniform_sampling(odl.Interval(0, 1), num_nodes=10, as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n / 2 + 1)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max point should be half a positive recip stride
    assert all_almost_equal(rgrid.max_pt, true_recip_stride / 2)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=True)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n / 2 + 1)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Max should be zero
    assert all_almost_equal(rgrid.max_pt, 0)


def test_reciprocal_nd():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift altogether
    rgrid = reciprocal(grid, shift=False, halfcomplex=False)

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)


def test_reciprocal_nd_shift_list():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)
    shift = [False, True, False]

    true_recip_stride = 2 * pi / (s * n)

    # Shift only the even dimension, then zero must be contained
    rgrid = reciprocal(grid, shift=shift, halfcomplex=False)
    noshift = np.where(np.logical_not(shift))

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt[noshift], -rgrid.max_pt[noshift])
    assert all_almost_equal(rgrid[n // 2], [0] * 3)


def test_reciprocal_nd_halfcomplex():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)
    stride_last = 2 * pi / (s[-1] * n[-1])
    n[-1] = n[-1] // 2 + 1

    # Without shift
    rgrid = reciprocal(grid, shift=False, halfcomplex=True)
    assert all_equal(rgrid.shape, n)
    assert rgrid.max_pt[-1] == 0  # last dim is odd

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=True)
    assert all_equal(rgrid.shape, n)
    assert rgrid.max_pt[-1] == -stride_last / 2


def test_dft_preproc_data():

    shape = (2, 3, 4)
    space_discr = odl.uniform_discr([0] * 3, [1] * 3, shape,
                                    field=odl.ComplexNumbers())

    # With shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * ((i + j + k) % 2))

    dfunc = space_discr.one()
    dft_preproc_data(dfunc, shift=True)

    assert all_almost_equal(dfunc.ntuple, correct_arr)

    # Without shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        argsum = sum((idx * (1 - 1 / shp))
                     for idx, shp in zip((i, j, k), shape))

        correct_arr.append(np.exp(1j * np.pi * argsum))

    dfunc = space_discr.one()
    dft_preproc_data(dfunc, shift=False)

    assert all_almost_equal(dfunc.ntuple, correct_arr)


def test_dft_postproc_data():

    shape = (2, 3, 4)
    space_discr = odl.uniform_discr([0] * 3, [1] * 3, shape,
                                    field=odl.ComplexNumbers())

    # With shift
    rgrid = reciprocal(space_discr.grid, shift=True)
    freq_cube = rgrid.convex_hull()
    recip_space_discr = odl.DiscreteLp(
        odl.FunctionSpace(freq_cube, field=odl.ComplexNumbers()),
        rgrid, odl.Cn(np.prod(shape)))

    correct_arr = []
    x0 = space_discr.grid.min_pt
    xi_0, rstride, nsamp = rgrid.min_pt, rgrid.stride, rgrid.shape
    for k in product(range(nsamp[0]), range(nsamp[1]), range(nsamp[2])):
        correct_arr.append(np.exp(-1j * np.dot(x0, xi_0 + rstride * k)))

    dfunc = recip_space_discr.one()
    dft_postproc_data(dfunc, x0)

    assert all_almost_equal(dfunc.ntuple, correct_arr)

    # Without shift
    rgrid = reciprocal(space_discr.grid, shift=False)
    freq_cube = rgrid.convex_hull()
    recip_space_discr = odl.DiscreteLp(
        odl.FunctionSpace(freq_cube, field=odl.ComplexNumbers()),
        rgrid, odl.Cn(np.prod(shape)))

    correct_arr = []
    x0 = space_discr.grid.min_pt
    xi_0, rstride, nsamp = rgrid.min_pt, rgrid.stride, rgrid.shape
    for k in product(range(nsamp[0]), range(nsamp[1]), range(nsamp[2])):
        correct_arr.append(np.exp(-1j * np.dot(x0, xi_0 + rstride * k)))

    dfunc = recip_space_discr.one()
    dft_postproc_data(dfunc, x0)

    assert all_almost_equal(dfunc.ntuple, correct_arr)


def test_dft_range(exponent, dtype):
    # Check if the range is initialized correctly. Encompasses the init test

    def conj(ex):
        if ex == 1.0:
            return float('inf')
        elif ex == float('inf'):
            return 1.0
        else:
            return ex / (ex - 1.0)

    # Testing R2C for real dtype, else C2C

    # 1D
    field = odl.RealNumbers() if is_real_dtype(dtype) else odl.ComplexNumbers()
    nsamp = 10
    space_discr = odl.uniform_discr(0, 1, nsamp, exponent=exponent,
                                    impl='numpy', dtype=dtype, field=field)

    dft = DiscreteFourierTransform(space_discr, halfcomplex=True, shift=True)
    assert dft.range.field == odl.ComplexNumbers()
    halfcomplex = True if is_real_dtype(dtype) else False
    assert dft.range.grid == reciprocal(dft.domain.grid,
                                        halfcomplex=halfcomplex,
                                        shift=True)
    assert dft.range.exponent == conj(exponent)

    # 3D
    field = odl.RealNumbers() if is_real_dtype(dtype) else odl.ComplexNumbers()
    nsamp = (3, 4, 5)
    space_discr = odl.uniform_discr([0] * 3, [1] * 3, nsamp, exponent=exponent,
                                    impl='numpy', dtype=dtype, field=field)

    dft = DiscreteFourierTransform(space_discr, halfcomplex=True, shift=True)
    assert dft.range.field == odl.ComplexNumbers()
    halfcomplex = True if is_real_dtype(dtype) else False
    assert dft.range.grid == reciprocal(dft.domain.grid,
                                        halfcomplex=halfcomplex,
                                        shift=True)
    assert dft.range.exponent == conj(exponent)


@skip_if_no_pyfftw
def test_dft_call(dtype):

    if is_real_dtype(dtype):
        field = odl.RealNumbers()
        halfcomplex = True
    else:
        field = odl.ComplexNumbers()
        halfcomplex = False

    # 1D
    nsamp = 10
    space_discr = odl.uniform_discr(0, 1, nsamp, exponent=2.0,
                                    impl='numpy', dtype=dtype, field=field)

    dfunc = space_discr.one()
    dft_arr = dft_call(dfunc.asarray(), halfcomplex=halfcomplex)
    # DFT of an array of ones equals nsamp in the first component and 0 else
    if halfcomplex:
        true_dft = [nsamp] + [0] * (nsamp // 2)
    else:
        true_dft = [nsamp] + [0] * (nsamp - 1)

    assert all_almost_equal(dft_arr, true_dft)

    # 3D
    nsamp = (3, 4, 5)
    space_discr = odl.uniform_discr([0] * 3, [1] * 3, nsamp, exponent=2.0,
                                    impl='numpy', dtype=dtype, field=field)

    dfunc = space_discr.one()
    dft_arr = dft_call(dfunc.asarray(), halfcomplex=halfcomplex)
    # DFT of an array of ones is size in the component (0, 0, 0) and 0 else
    if halfcomplex:
        shape = list(nsamp)
        shape[-1] = nsamp[-1] // 2 + 1
    else:
        shape = nsamp
    true_dft = np.zeros(shape)
    true_dft[0, 0, 0] = dfunc.space.grid.size

    print('dft: ', dft_arr)
    print('true: ', true_dft)
    assert all_almost_equal(dft_arr, true_dft)


@skip_if_no_pyfftw
def test_dft_plan(planning):

    # 1D, C2C
    nsamp = 10
    space_discr = odl.uniform_discr(0, 1, nsamp, exponent=2.0, impl='numpy',
                                    field=odl.ComplexNumbers())

    dfunc = space_discr.one()
    dft_arr = dft_call(dfunc.asarray(), planning=planning)
    twice_dft_arr = dft_call(dft_arr)
    # Unnormalized DFT, should give the mirrored array x number of samples
    assert all_almost_equal(twice_dft_arr[::-1],
                            dfunc.space.grid.size * dfunc.ntuple)

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
