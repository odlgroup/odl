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
    reciprocal, dft_preprocess_data, dft_postprocess_data,
    pyfftw_call, DiscreteFourierTransform, _TYPE_MAP_R2C)
from odl.util.testutils import all_almost_equal, all_equal
from odl.util.utility import is_real_dtype


# TODO: add reciprocal test with axes
pytestmark = pytest.mark.skipif("not odl.trafos.PYFFTW_AVAILABLE")


exp_params = [2.0, 1.0, float('inf'), 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]
exp_fixture = pytest.fixture(scope="module", ids=exp_ids, params=exp_params)


@exp_fixture
def exponent(request):
    return request.param


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


plan_params = ['estimate', 'measure', 'patient', 'exhaustive']
plan_ids = [" planning = '{}' ".format(p) for p in plan_params]
plan_fixture = pytest.fixture(scope="module", ids=plan_ids,
                              params=plan_params)


@plan_fixture
def planning(request):
    return request.param


def _random_array(shape, dtype):
    if is_real_dtype(dtype):
        return np.random.rand(*shape).astype(dtype)
    else:
        return (np.random.rand(*shape).astype(dtype) +
                1j * np.random.rand(*shape).astype(dtype))


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


def test_reciprocal_nd_axes():

    cube = odl.Cuboid([0] * 3, [1] * 3)
    grid = odl.uniform_sampling(cube, num_nodes=(3, 4, 5), as_midp=True)
    s = grid.stride
    n = np.array(grid.shape)
    axes_list = [[1, -1], [0], [0, 2, 1], [2, 0]]

    for axes in axes_list:
        active = np.zeros(grid.ndim, dtype=bool)
        active[axes] = True
        inactive = np.logical_not(active)

        true_recip_stride = np.empty(grid.ndim)
        true_recip_stride[active] = 2 * pi / (s[active] * n[active])
        true_recip_stride[inactive] = s[inactive]

        # Without shift altogether
        rgrid = reciprocal(grid, shift=False, axes=axes, halfcomplex=False)

        assert all_equal(rgrid.shape, n)
        assert all_almost_equal(rgrid.stride, true_recip_stride)
        assert all_almost_equal(rgrid.min_pt[active], -rgrid.max_pt[active])
        assert all_equal(rgrid.min_pt[inactive], grid.min_pt[inactive])
        assert all_equal(rgrid.max_pt[inactive], grid.max_pt[inactive])


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


def test_dft_preprocess_data():

    shape = (2, 3, 4)
    space_discr = odl.uniform_discr([0] * 3, [1] * 3, shape,
                                    field=odl.ComplexNumbers())

    # With shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * ((i + j + k) % 2))

    dfunc = space_discr.one()
    dft_preprocess_data(dfunc, shift=True)

    assert all_almost_equal(dfunc.ntuple, correct_arr)

    # Without shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        argsum = sum((idx * (1 - 1 / shp))
                     for idx, shp in zip((i, j, k), shape))

        correct_arr.append(np.exp(1j * np.pi * argsum))

    dfunc = space_discr.one()
    dft_preprocess_data(dfunc, shift=False)

    assert all_almost_equal(dfunc.ntuple, correct_arr)


@pytest.mark.xfail(reason='Function rewritten, new test necessary')
def test_dft_postprocess_data():

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
    xi_0, rstride, shape = rgrid.min_pt, rgrid.stride, rgrid.shape
    for k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(np.exp(-1j * np.dot(x0, xi_0 + rstride * k)))

    dfunc = recip_space_discr.one()
    dft_postprocess_data(dfunc, x0)

    assert all_almost_equal(dfunc.ntuple, correct_arr)

    # Without shift
    rgrid = reciprocal(space_discr.grid, shift=False)
    freq_cube = rgrid.convex_hull()
    recip_space_discr = odl.DiscreteLp(
        odl.FunctionSpace(freq_cube, field=odl.ComplexNumbers()),
        rgrid, odl.Cn(np.prod(shape)))

    correct_arr = []
    x0 = space_discr.grid.min_pt
    xi_0, rstride, shape = rgrid.min_pt, rgrid.stride, rgrid.shape
    for k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(np.exp(-1j * np.dot(x0, xi_0 + rstride * k)))

    dfunc = recip_space_discr.one()
    dft_postprocess_data(dfunc, x0)

    assert all_almost_equal(dfunc.ntuple, correct_arr)


def _params_from_dtype(dt):
    if is_real_dtype(dt):
        halfcomplex = True
        dtype = _TYPE_MAP_R2C[np.dtype(dt)]
    else:
        halfcomplex = False
        dtype = dt
    return halfcomplex, dtype


def _halfcomplex_shape(shape, axes=None):
    if axes is None:
        axes = tuple(range(len(shape)))

    shape = list(shape)
    shape[axes[-1]] = shape[axes[-1]] // 2 + 1
    return shape


def test_pyfftw_call_forward(dtype):
    # Test against Numpy's FFT
    halfcomplex, out_dtype = _params_from_dtype(dtype)

    for shape in [(10,), (3, 4, 5)]:
        arr = _random_array(shape, dtype)

        if halfcomplex:
            true_dft = np.fft.rfftn(arr)
            dft_arr = np.empty(_halfcomplex_shape(shape), dtype=out_dtype)
        else:
            true_dft = np.fft.fftn(arr)
            dft_arr = np.empty(shape, dtype=out_dtype)

        pyfftw_call(arr, dft_arr, direction='forward',
                    halfcomplex=halfcomplex, preserve_input=False)

        assert all_almost_equal(dft_arr, true_dft)


def test_pyfftw_call_backward(dtype):
    # Test against Numpy's IFFT, no normalization
    halfcomplex, in_dtype = _params_from_dtype(dtype)

    for shape in [(10,), (3, 4, 5)]:
        # Scaling happens wrt output (large) shape
        idft_scaling = np.prod(shape)

        if halfcomplex:
            arr = _random_array(_halfcomplex_shape(shape), in_dtype)
            true_idft = np.fft.irfftn(arr, shape) * idft_scaling
        else:
            arr = _random_array(shape, in_dtype)
            true_idft = np.fft.ifftn(arr) * idft_scaling

        idft_arr = np.empty(shape, dtype=dtype)
        pyfftw_call(arr, idft_arr, direction='backward',
                    halfcomplex=halfcomplex)

        assert all_almost_equal(idft_arr, true_idft)


def test_pyfftw_call_forward_bad_input():

    # Complex

    # Bad dtype
    dtype_in = 'complex128'
    arr_in = np.empty(3, dtype=dtype_in)
    bad_dtypes_out = ['complex64', 'float64', 'float128']
    for bad_dtype in bad_dtypes_out:
        arr_out = np.empty(3, dtype=bad_dtype)
        with pytest.raises(TypeError):
            pyfftw_call(arr_in, arr_out, halfcomplex=False)

    # Bad shape
    shape = (3, 4)
    arr_in = np.empty(shape, dtype='complex128')
    bad_shapes_out = [(3, 3), (3,), (4,), (3, 4, 5), ()]
    for bad_shape in bad_shapes_out:
        arr_out = np.empty(bad_shape, dtype='complex128')
        with pytest.raises(ValueError):
            pyfftw_call(arr_in, arr_out, halfcomplex=False)

    # Duplicate axes
    arr_in = np.empty((3, 4, 5), dtype='complex128')
    arr_out = np.empty_like(arr_in)
    bad_axes_list = [(0, 0, 1), (1, 1, 1), (-1, -1)]
    for bad_axes in bad_axes_list:
        with pytest.raises(ValueError):
            pyfftw_call(arr_in, arr_out, axes=bad_axes)

    # Halfcomplex

    # Bad dtype
    dtype_in = 'float64'
    arr_in = np.empty(10, dtype=dtype_in)
    bad_dtypes_out = ['complex64', 'float64', 'float128', 'complex256']
    for bad_dtype in bad_dtypes_out:
        arr_out = np.empty(6, dtype=bad_dtype)
        with pytest.raises(TypeError):
            pyfftw_call(arr_in, arr_out, halfcomplex=True)

    # Bad shape
    shape = (3, 4, 5)
    arr_in = np.empty(shape, dtype='float64')
    axes_list = [None, (0, 1), (1,), (1, 2), (2, 1), (-1, -2, -3)]
    # Correct shapes:
    # [(3, 4, 3), (3, 3, 5), (3, 3, 5), (3, 4, 3), (3, 3, 5), (2, 4, 5)]
    bad_shapes_out = [(3, 4, 2), (3, 4, 3), (2, 3, 5), (3, 2, 3), (3, 4, 3),
                      (3, 4, 3)]
    always_bad_shapes = [(3, 4), (3, 4, 5)]
    for bad_shape, axes in zip(bad_shapes_out, axes_list):

        for always_bad_shape in always_bad_shapes:
            arr_out = np.empty(always_bad_shape, dtype='complex128')
            with pytest.raises(ValueError):
                pyfftw_call(arr_in, arr_out, axes=axes, halfcomplex=True)

        arr_out = np.empty(bad_shape, dtype='complex128')
        with pytest.raises(ValueError):
            pyfftw_call(arr_in, arr_out, axes=axes, halfcomplex=True)


def test_pyfftw_call_forward_real_not_halfcomplex():
    # Test against Numpy's FFT
    for shape in [(10,), (3, 4, 5)]:
        arr = _random_array(shape, dtype='float64')

        true_dft = np.fft.fftn(arr)
        dft_arr = np.empty(shape, dtype='complex128')
        pyfftw_call(arr, dft_arr, direction='forward', halfcomplex=False)

        assert all_almost_equal(dft_arr, true_dft)


def test_pyfftw_call_backward_real_not_halfcomplex():
    # Test against Numpy's IFFT, no normalization
    for shape in [(10,), (3, 4, 5)]:
        # Scaling happens wrt output (large) shape
        idft_scaling = np.prod(shape)

        arr = _random_array(shape, dtype='float64')
        true_idft = np.fft.ifftn(arr) * idft_scaling
        idft_arr = np.empty(shape, dtype='complex128')
        pyfftw_call(arr, idft_arr, direction='backward', halfcomplex=False)

        assert all_almost_equal(idft_arr, true_idft)


def test_pyfftw_plan_preserve_input(planning):

    for shape in [(10,), (3, 4)]:
        arr = _random_array(shape, dtype='complex128')
        arr_cpy = arr.copy()

        idft_scaling = np.prod(shape)
        true_idft = np.fft.ifftn(arr) * idft_scaling
        idft_arr = np.empty(shape, dtype='complex128')
        pyfftw_call(arr, idft_arr, direction='backward', halfcomplex=False,
                    planning=planning)

        assert all_almost_equal(arr, arr_cpy)  # Input perserved
        assert all_almost_equal(idft_arr, true_idft)

        pyfftw_call(arr, idft_arr, direction='backward', halfcomplex=False,
                    planning=planning)

        assert all_almost_equal(idft_arr, true_idft)


def test_pyfftw_call_forward_with_axes(dtype):

    halfcomplex, out_dtype = _params_from_dtype(dtype)
    shape = (3, 4, 5)

    test_axes = [(0, 1), (1,), (-1,), (1, 0), (-1, -2, -3)]
    for axes in test_axes:
        arr = _random_array(shape, dtype)
        if halfcomplex:
            true_dft = np.fft.rfftn(arr, axes=axes)
            dft_arr = np.empty(_halfcomplex_shape(shape, axes),
                               dtype=out_dtype)
        else:
            true_dft = np.fft.fftn(arr, axes=axes)
            dft_arr = np.empty(shape, dtype=out_dtype)

        pyfftw_call(arr, dft_arr, direction='forward', axes=axes,
                    halfcomplex=halfcomplex)

        assert all_almost_equal(dft_arr, true_dft)


def test_pyfftw_call_backward_with_axes(dtype):

    halfcomplex, in_dtype = _params_from_dtype(dtype)
    shape = (3, 4, 5)

    test_axes = [(0, 1), (1,), (-1,), (1, 0), (-1, -2, -3)]
    for axes in test_axes:
        # Only the shape indexed by axes count for the scaling
        active_shape = np.take(shape, axes)
        idft_scaling = np.prod(active_shape)

        if halfcomplex:
            arr = _random_array(_halfcomplex_shape(shape, axes), in_dtype)
            true_idft = (np.fft.irfftn(arr, s=active_shape, axes=axes) *
                         idft_scaling)
        else:
            arr = _random_array(shape, in_dtype)
            true_idft = (np.fft.ifftn(arr, s=active_shape, axes=axes) *
                         idft_scaling)

        idft_arr = np.empty(shape, dtype=dtype)
        pyfftw_call(arr, idft_arr, direction='backward', axes=axes,
                    halfcomplex=halfcomplex)

        assert all_almost_equal(idft_arr, true_idft)


def test_pyfftw_call_forward_with_plan():

    for shape in [(10,), (3, 4, 5)]:
        arr = _random_array(shape, dtype='complex128')
        arr_cpy = arr.copy()
        true_dft = np.fft.fftn(arr)

        # First run, create plan
        dft_arr = np.empty(shape, dtype='complex128')
        plan = pyfftw_call(arr, dft_arr, direction='forward',
                           halfcomplex=False, planning_effort='measure')

        # Second run, reuse with fresh output array
        dft_arr = np.empty(shape, dtype='complex128')
        pyfftw_call(arr, dft_arr, direction='forward', fftw_plan=plan,
                    halfcomplex=False)

        assert all_almost_equal(arr, arr_cpy)  # Input perserved
        assert all_almost_equal(dft_arr, true_dft)


def test_pyfftw_call_backward_with_plan():

    for shape in [(10,), (3, 4, 5)]:
        arr = _random_array(shape, dtype='complex128')
        arr_cpy = arr.copy()
        idft_scaling = np.prod(shape)
        true_idft = np.fft.ifftn(arr) * idft_scaling

        # First run, create plan
        idft_arr = np.empty(shape, dtype='complex128')
        plan = pyfftw_call(arr, idft_arr, direction='backward',
                           halfcomplex=False, planning_effort='measure')

        # Second run, reuse with fresh output array
        idft_arr = np.empty(shape, dtype='complex128')
        pyfftw_call(arr, idft_arr, direction='backward', fftw_plan=plan,
                    halfcomplex=False)

        assert all_almost_equal(arr, arr_cpy)  # Input perserved
        assert all_almost_equal(idft_arr, true_idft)


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
    shape = 10
    space_discr = odl.uniform_discr(0, 1, shape, exponent=exponent,
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
    shape = (3, 4, 5)
    space_discr = odl.uniform_discr([0] * 3, [1] * 3, shape, exponent=exponent,
                                    impl='numpy', dtype=dtype, field=field)

    dft = DiscreteFourierTransform(space_discr, halfcomplex=True, shift=True)
    assert dft.range.field == odl.ComplexNumbers()
    halfcomplex = True if is_real_dtype(dtype) else False
    assert dft.range.grid == reciprocal(dft.domain.grid,
                                        halfcomplex=halfcomplex,
                                        shift=True)
    assert dft.range.exponent == conj(exponent)


@pytest.mark.xfail(reason='Wrong scaling and shift issue')
def test_dft_with_known_functions():

    # 1d
    # Characteristic function of [0, 1], its Fourier transform is
    # given by 1j * (exp(-1j * y) - 1) / y
    def char_interval(x):
        return np.where(np.logical_and(x >= 0, x <= 1), 1.0, 0.0)

    def char_interval_ft(x):
        def _at_small_args(x):
            return 1 - 1j * x / 2 + x ** 2 / 6

        def _at_large_args(x):
            return (1 - np.exp(-1j * x)) / (1j * x)

        return np.where(np.abs(x) > 1e-5, _at_large_args(x),
                        _at_small_args(x)) / np.sqrt(2 * np.pi)

    discr = odl.uniform_discr(-2, 2, 64, impl='numpy')
    dft = DiscreteFourierTransform(discr)
    func = dft.domain.element(char_interval)
    func_true_ft = dft.range.element(char_interval_ft)
    func_dft = dft(func)
    assert all_almost_equal(func_dft, func_true_ft)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
