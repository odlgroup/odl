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
import pytest

# ODL imports
import odl
from odl.trafos.fourier import (
    reciprocal, inverse_reciprocal, dft_preprocess_data, dft_postprocess_data,
    pyfftw_call, _interp_kernel_ft,
    DiscreteFourierTransform, DiscreteFourierTransformInverse,
    FourierTransform)
from odl.util import (all_almost_equal, all_equal,
                      never_skip, skip_if_no_pyfftw,
                      is_real_dtype, conj_exponent, TYPE_MAP_R2C)


exp_params = [2.0, 1.0, float('inf'), 1.5]
exp_ids = [' p = {} '.format(p) for p in exp_params]


@pytest.fixture(scope="module", ids=exp_ids, params=exp_params)
def exponent(request):
    return request.param


dtype_params = [str(dtype) for dtype in TYPE_MAP_R2C.keys()]
dtype_params += [str(dtype) for dtype in TYPE_MAP_R2C.values()]
dtype_params = list(set(dtype_params))
dtype_ids = [' dtype = {} '.format(dt) for dt in dtype_params]


@pytest.fixture(scope="module", ids=dtype_ids, params=dtype_params)
def dtype(request):
    return request.param


plan_params = ['estimate', 'measure', 'patient', 'exhaustive']
plan_ids = [" planning = '{}' ".format(p) for p in plan_params]


@pytest.fixture(scope="module", ids=plan_ids, params=plan_params)
def planning(request):
    return request.param


impl_params = [never_skip('numpy'), skip_if_no_pyfftw('pyfftw')]
impl_ids = ['impl={}'.format(impl.args[1]) for impl in impl_params]


@pytest.fixture(scope="module", ids=impl_ids, params=impl_params)
def impl(request):
    return request.param


@pytest.fixture(scope='module', ids=[' - ', ' + '], params=['-', '+'])
def sign(request):
    return request.param


def _random_array(shape, dtype):
    if is_real_dtype(dtype):
        return np.random.rand(*shape).astype(dtype)
    else:
        return (np.random.rand(*shape).astype(dtype) +
                1j * np.random.rand(*shape).astype(dtype))


@pytest.fixture(scope='module', ids=[' forward ', ' backward '],
                params=['forward', 'backward'])
def direction(request):
    return request.param


# ---- reciprocal ---- #


def test_reciprocal_1d_odd():

    grid = odl.uniform_sampling(0, 1, shape=11)
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
    assert all_almost_equal(rgrid.mid_pt, 0)
    # Zero should be at index n // 2
    assert all_almost_equal(rgrid[n // 2], 0)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # No point should be closer to 0 than half a recip stride
    atol = 0.999 * true_recip_stride / 2
    assert not rgrid.approx_contains(0, atol=atol)

    # Inverting the reciprocal should give back the original
    irgrid = inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=False)
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_1d_odd_halfcomplex():

    grid = odl.uniform_sampling(0, 1, shape=11)
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

    # Inverting the reciprocal should give back the original
    irgrid = inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=True,
                                halfcx_parity='odd')
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_1d_even():

    grid = odl.uniform_sampling(0, 1, shape=10)
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
    assert all_almost_equal(rgrid.mid_pt, 0)
    # No point should be closer to 0 than half a recip stride
    atol = 0.999 * true_recip_stride / 2
    assert not rgrid.approx_contains(0, atol=atol)

    # With shift
    rgrid = reciprocal(grid, shift=True, halfcomplex=False)

    # Independent of shift and halfcomplex, check anyway
    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    # Zero should be at index n // 2
    assert all_almost_equal(rgrid[n // 2], 0)

    # Inverting the reciprocal should give back the original
    irgrid = inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=False)
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_1d_even_halfcomplex():

    grid = odl.uniform_sampling(0, 1, shape=10)
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

    # Inverting the reciprocal should give back the original
    irgrid = inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=True,
                                halfcx_parity='even')
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_nd():

    grid = odl.uniform_sampling([0] * 3, [1] * 3, shape=(3, 4, 5))
    s = grid.stride
    n = np.array(grid.shape)

    true_recip_stride = 2 * pi / (s * n)

    # Without shift altogether
    rgrid = reciprocal(grid, shift=False, halfcomplex=False)

    assert all_equal(rgrid.shape, n)
    assert all_almost_equal(rgrid.stride, true_recip_stride)
    assert all_almost_equal(rgrid.min_pt, -rgrid.max_pt)

    # Inverting the reciprocal should give back the original
    irgrid = inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=False)
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_nd_shift_list():

    grid = odl.uniform_sampling([0] * 3, [1] * 3, shape=(3, 4, 5))
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

    # Inverting the reciprocal should give back the original
    irgrid = inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=False)
    assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_nd_axes():

    grid = odl.uniform_sampling([0] * 3, [1] * 3, shape=(3, 4, 5))
    s = grid.stride
    n = np.array(grid.shape)
    axes_list = [[1, -1], [0], 0, [0, 2, 1], [2, 0]]

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

        # Inverting the reciprocal should give back the original
        irgrid = inverse_reciprocal(rgrid, grid.min_pt, axes=axes,
                                    halfcomplex=False)
        assert irgrid.approx_equals(grid, atol=1e-6)


def test_reciprocal_nd_halfcomplex():

    grid = odl.uniform_sampling([0] * 3, [1] * 3, shape=(3, 4, 5))
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

    # Inverting the reciprocal should give back the original
    irgrid = inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=True,
                                halfcx_parity='odd')
    assert irgrid.approx_equals(grid, atol=1e-6)

    with pytest.raises(ValueError):
        inverse_reciprocal(rgrid, grid.min_pt, halfcomplex=True,
                           halfcx_parity='+')

# ---- dft_preprocess_data ---- #


def test_dft_preprocess_data(sign):

    shape = (2, 3, 4)

    # With shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append((1 + 1j) * (1 - 2 * ((i + j + k) % 2)))

    arr = np.ones(shape, dtype='complex64') * (1 + 1j)
    preproc = dft_preprocess_data(arr, shift=True, sign=sign)  # out-of-place
    dft_preprocess_data(arr, shift=True, out=arr, sign=sign)  # in-place

    assert all_almost_equal(preproc.ravel(), correct_arr)
    assert all_almost_equal(arr.ravel(), correct_arr)

    # Without shift
    imag = 1j if sign == '-' else -1j
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        argsum = sum((idx * (1 - 1 / shp))
                     for idx, shp in zip((i, j, k), shape))

        correct_arr.append((1 + 1j) * np.exp(imag * np.pi * argsum))

    arr = np.ones(shape, dtype='complex64') * (1 + 1j)
    dft_preprocess_data(arr, shift=False, out=arr, sign=sign)

    assert all_almost_equal(arr.ravel(), correct_arr)

    # Bad input
    with pytest.raises(ValueError):
        dft_preprocess_data(arr, out=arr, sign=1)

    arr = np.zeros(shape, dtype='S2')
    with pytest.raises(ValueError):
        dft_preprocess_data(arr)


def test_dft_preprocess_data_halfcomplex(sign):

    shape = (2, 3, 4)

    # With shift
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * ((i + j + k) % 2))

    arr = np.ones(shape, dtype='float64')
    preproc = dft_preprocess_data(arr, shift=True, sign=sign)  # out-of-place
    out = np.empty_like(arr)
    dft_preprocess_data(arr, shift=True, out=out, sign=sign)  # in-place
    dft_preprocess_data(arr, shift=True, out=arr, sign=sign)  # in-place
    assert all_almost_equal(preproc.ravel(), correct_arr)
    assert all_almost_equal(arr.ravel(), correct_arr)
    assert all_almost_equal(out.ravel(), correct_arr)

    # Without shift
    imag = 1j if sign == '-' else -1j
    correct_arr = []
    for i, j, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        argsum = sum((idx * (1 - 1 / shp))
                     for idx, shp in zip((i, j, k), shape))

        correct_arr.append(np.exp(imag * np.pi * argsum))

    arr = np.ones(shape, dtype='float64')
    preproc = dft_preprocess_data(arr, shift=False, sign=sign)
    assert all_almost_equal(preproc.ravel(), correct_arr)

    # Non-float input works, too
    arr = np.ones(shape, dtype='int')
    preproc = dft_preprocess_data(arr, shift=False, sign=sign)
    assert all_almost_equal(preproc.ravel(), correct_arr)

    # In-place modification not possible for float array and no shift
    arr = np.ones(shape, dtype='float64')
    with pytest.raises(ValueError):
        dft_preprocess_data(arr, shift=False, out=arr, sign=sign)


def test_dft_preprocess_data_with_axes(sign):

    shape = (2, 3, 4)

    axes = 1  # Only middle index counts
    correct_arr = []
    for _, j, __ in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * (j % 2))

    arr = np.ones(shape, dtype='complex64')
    dft_preprocess_data(arr, shift=True, axes=axes, out=arr, sign=sign)

    assert all_almost_equal(arr.ravel(), correct_arr)

    axes = [0, -1]  # First and last
    correct_arr = []
    for i, _, k in product(range(shape[0]), range(shape[1]), range(shape[2])):
        correct_arr.append(1 - 2 * ((i + k) % 2))

    arr = np.ones(shape, dtype='complex64')
    dft_preprocess_data(arr, shift=True, axes=axes, out=arr, sign=sign)

    assert all_almost_equal(arr.ravel(), correct_arr)


# ---- pyfftw_call ---- #


def _params_from_dtype(dt):
    if is_real_dtype(dt):
        halfcomplex = True
        dtype = TYPE_MAP_R2C[np.dtype(dt)]
    else:
        halfcomplex = False
        dtype = dt
    return halfcomplex, dtype


def _halfcomplex_shape(shape, axes=None):
    if axes is None:
        axes = tuple(range(len(shape)))

    try:
        axes = (int(axes),)
    except TypeError:
        pass

    shape = list(shape)
    shape[axes[-1]] = shape[axes[-1]] // 2 + 1
    return shape


@skip_if_no_pyfftw
def test_pyfftw_call_forward(dtype):
    # Test against Numpy's FFT
    if dtype == np.dtype('float16'):  # not supported, skipping
        return

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


@skip_if_no_pyfftw
def test_pyfftw_call_threads():
    shape = (3, 4, 5)
    arr = _random_array(shape, dtype='complex64')
    true_dft = np.fft.fftn(arr)
    dft_arr = np.empty(shape, dtype='complex64')
    pyfftw_call(arr, dft_arr, direction='forward', preserve_input=False,
                threads=4)
    assert all_almost_equal(dft_arr, true_dft)

    shape = (1000,)  # Trigger cpu_count() as number of threads
    arr = _random_array(shape, dtype='complex64')
    true_dft = np.fft.fftn(arr)
    dft_arr = np.empty(shape, dtype='complex64')
    pyfftw_call(arr, dft_arr, direction='forward', preserve_input=False)

    assert all_almost_equal(dft_arr, true_dft)


@skip_if_no_pyfftw
def test_pyfftw_call_backward(dtype):
    # Test against Numpy's IFFT, no normalization
    if dtype == np.dtype('float16'):  # not supported, skipping
        return

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


@skip_if_no_pyfftw
def test_pyfftw_call_bad_input(direction):

    # Complex

    # Bad dtype
    dtype_in = 'complex128'
    arr_in = np.empty(3, dtype=dtype_in)
    bad_dtypes_out = np.sctypes['float'] + np.sctypes['complex']
    try:
        # This one is correct, so we remove it
        bad_dtypes_out.remove(np.dtype('complex128'))
    except ValueError:
        pass
    for bad_dtype in bad_dtypes_out:
        arr_out = np.empty(3, dtype=bad_dtype)
        with pytest.raises(ValueError):
            pyfftw_call(arr_in, arr_out, halfcomplex=False,
                        direction=direction)

    # Bad shape
    shape = (3, 4)
    arr_in = np.empty(shape, dtype='complex128')
    bad_shapes_out = [(3, 3), (3,), (4,), (3, 4, 5), ()]
    for bad_shape in bad_shapes_out:
        arr_out = np.empty(bad_shape, dtype='complex128')
        with pytest.raises(ValueError):
            pyfftw_call(arr_in, arr_out, halfcomplex=False,
                        direction=direction)

    # Duplicate axes
    arr_in = np.empty((3, 4, 5), dtype='complex128')
    arr_out = np.empty_like(arr_in)
    bad_axes_list = [(0, 0, 1), (1, 1, 1), (-1, -1)]
    for bad_axes in bad_axes_list:
        with pytest.raises(ValueError):
            pyfftw_call(arr_in, arr_out, axes=bad_axes,
                        direction=direction)

    # Axis entry out of range
    arr_in = np.empty((3, 4, 5), dtype='complex128')
    arr_out = np.empty_like(arr_in)
    bad_axes_list = [(0, 3), (-4,)]
    for bad_axes in bad_axes_list:
        with pytest.raises(IndexError):
            pyfftw_call(arr_in, arr_out, axes=bad_axes,
                        direction=direction)

    # Halfcomplex not possible for complex data
    arr_in = np.empty((3, 4, 5), dtype='complex128')
    arr_out = np.empty_like(arr_in)
    with pytest.raises(ValueError):
        pyfftw_call(arr_in, arr_out, halfcomplex=True,
                    direction=direction)

    # Data type mismatch
    arr_in = np.empty((3, 4, 5), dtype='complex128')
    arr_out = np.empty_like(arr_in, dtype='complex64')
    with pytest.raises(ValueError):
        pyfftw_call(arr_in, arr_out, direction=direction)

    # Halfcomplex

    # Bad dtype
    dtype_in = 'float64'
    arr_in = np.empty(10, dtype=dtype_in)
    bad_dtypes_out = np.sctypes['float'] + np.sctypes['complex']
    try:
        # This one is correct, so we remove it
        bad_dtypes_out.remove(np.dtype('complex128'))
    except ValueError:
        pass
    for bad_dtype in bad_dtypes_out:
        arr_out = np.empty(6, dtype=bad_dtype)
        with pytest.raises(ValueError):
            if direction == 'forward':
                pyfftw_call(arr_in, arr_out, halfcomplex=True,
                            direction='forward')
            else:
                pyfftw_call(arr_out, arr_in, halfcomplex=True,
                            direction='backward')

    # Bad shape
    shape = (3, 4, 5)
    axes_list = [None, (0, 1), (1,), (1, 2), (2, 1), (-1, -2, -3)]
    arr_in = np.empty(shape, dtype='float64')
    # Correct shapes:
    # [(3, 4, 3), (3, 3, 5), (3, 3, 5), (3, 4, 3), (3, 3, 5), (2, 4, 5)]
    bad_shapes_out = [(3, 4, 2), (3, 4, 3), (2, 3, 5), (3, 2, 3),
                      (3, 4, 3), (3, 4, 3)]
    always_bad_shapes = [(3, 4), (3, 4, 5)]
    for bad_shape, axes in zip(bad_shapes_out, axes_list):

        for always_bad_shape in always_bad_shapes:
            arr_out = np.empty(always_bad_shape, dtype='complex128')
            with pytest.raises(ValueError):
                if direction == 'forward':
                    pyfftw_call(arr_in, arr_out, axes=axes, halfcomplex=True,
                                direction='forward')
                else:
                    pyfftw_call(arr_out, arr_in, axes=axes, halfcomplex=True,
                                direction='backward')

        arr_out = np.empty(bad_shape, dtype='complex128')
        with pytest.raises(ValueError):
            if direction == 'forward':
                pyfftw_call(arr_in, arr_out, axes=axes, halfcomplex=True,
                            direction='forward')
            else:
                pyfftw_call(arr_out, arr_in, axes=axes, halfcomplex=True,
                            direction='backward')


@skip_if_no_pyfftw
def test_pyfftw_call_forward_real_not_halfcomplex():
    # Test against Numpy's FFT
    for shape in [(10,), (3, 4, 5)]:
        arr = _random_array(shape, dtype='float64')

        true_dft = np.fft.fftn(arr)
        dft_arr = np.empty(shape, dtype='complex128')
        pyfftw_call(arr, dft_arr, direction='forward', halfcomplex=False)

        assert all_almost_equal(dft_arr, true_dft)


@skip_if_no_pyfftw
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


@skip_if_no_pyfftw
def test_pyfftw_call_plan_preserve_input(planning):

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


@skip_if_no_pyfftw
def test_pyfftw_call_forward_with_axes(dtype):
    if dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, out_dtype = _params_from_dtype(dtype)
    shape = (3, 4, 5)

    test_axes = [(0, 1), [1], (-1,), (1, 0), (-1, -2, -3)]
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


@skip_if_no_pyfftw
def test_pyfftw_call_backward_with_axes(dtype):
    if dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, in_dtype = _params_from_dtype(dtype)
    shape = (3, 4, 5)

    test_axes = [(0, 1), [1], (-1,), (1, 0), (-1, -2, -3)]
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


@skip_if_no_pyfftw
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


@skip_if_no_pyfftw
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


# ---- DiscreteFourierTransform ---- #


def test_dft_init(impl):
    # Just check if the code runs at all
    shape = (4, 5)
    dom = odl.discr_sequence_space(shape)
    dom_nonseq = odl.uniform_discr([0, 0], [1, 1], shape)
    dom_f32 = odl.discr_sequence_space(shape, dtype='float32')
    ran = odl.discr_sequence_space(shape, dtype='complex128')
    ran_c64 = odl.discr_sequence_space(shape, dtype='complex64')
    ran_hc = odl.discr_sequence_space((3, 5), dtype='complex128')

    # Implicit range
    DiscreteFourierTransform(dom, impl=impl)
    DiscreteFourierTransform(dom_nonseq, impl=impl)
    DiscreteFourierTransform(dom_f32, impl=impl)
    DiscreteFourierTransform(dom, axes=(0,), impl=impl)
    DiscreteFourierTransform(dom, axes=(0, -1), impl=impl)
    DiscreteFourierTransform(dom, axes=(0,), halfcomplex=True, impl=impl)
    DiscreteFourierTransform(dom, impl=impl, sign='+')

    # Explicit range
    DiscreteFourierTransform(dom, range=ran, impl=impl)
    DiscreteFourierTransform(dom_f32, range=ran_c64, impl=impl)
    DiscreteFourierTransform(dom, range=ran, axes=(0,), impl=impl)
    DiscreteFourierTransform(dom, range=ran, axes=(0,), impl=impl, sign='+')
    DiscreteFourierTransform(dom, range=ran, axes=(0, -1), impl=impl)
    DiscreteFourierTransform(dom, range=ran_hc, axes=(0,), impl=impl,
                             halfcomplex=True)


def test_dft_init_raise():
    # Test different error scenarios
    shape = (4, 5)
    dom = odl.discr_sequence_space(shape)
    dom_f32 = odl.discr_sequence_space(shape, dtype='float32')

    # Bad types
    with pytest.raises(TypeError):
        DiscreteFourierTransform(dom.dspace)

    with pytest.raises(TypeError):
        DiscreteFourierTransform(dom, dom.dspace)

    # Illegal arguments
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, impl='fftw')

    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, axes=(1, 2))

    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, axes=(1, -3))

    # Badly shaped range
    bad_ran = odl.discr_sequence_space((3, 5), dtype='complex128')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, bad_ran)

    bad_ran = odl.discr_sequence_space((10, 10), dtype='complex128')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, bad_ran)

    bad_ran = odl.discr_sequence_space((4, 5), dtype='complex128')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, bad_ran, halfcomplex=True)

    bad_ran = odl.discr_sequence_space((4, 3), dtype='complex128')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, bad_ran, halfcomplex=True, axes=(0,))

    # Bad data types
    bad_ran = odl.discr_sequence_space(shape, dtype='complex64')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, bad_ran)

    bad_ran = odl.discr_sequence_space(shape, dtype='float64')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, bad_ran)

    bad_ran = odl.discr_sequence_space((4, 3), dtype='float64')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, bad_ran, halfcomplex=True)

    bad_ran = odl.discr_sequence_space((4, 3), dtype='complex128')
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom_f32, bad_ran, halfcomplex=True)

    # Bad sign
    with pytest.raises(ValueError):
        DiscreteFourierTransform(dom, sign=-1)


def test_dft_range():
    # 1d
    shape = 10
    dom = odl.discr_sequence_space(shape, dtype='complex128')
    fft = DiscreteFourierTransform(dom)
    true_ran = odl.discr_sequence_space(shape, dtype='complex128')
    assert fft.range == true_ran

    # 3d
    shape = (3, 4, 5)
    ran = odl.discr_sequence_space(shape, dtype='complex64')
    fft = DiscreteFourierTransform(ran)
    true_ran = odl.discr_sequence_space(shape, dtype='complex64')
    assert fft.range == true_ran

    # 3d, with axes and halfcomplex
    shape = (3, 4, 5)
    axes = (-1, -2)
    ran_shape = (3, 3, 5)
    dom = odl.discr_sequence_space(shape, dtype='float32')
    fft = DiscreteFourierTransform(dom, axes=axes, halfcomplex=True)
    true_ran = odl.discr_sequence_space(ran_shape, dtype='complex64')
    assert fft.range == true_ran


# ---- DiscreteFourierTransformInverse ---- #


def test_idft_init(impl):
    # Just check if the code runs at all; this uses the init function of
    # DiscreteFourierTransform, so we don't need exhaustive tests here
    shape = (4, 5)
    ran = odl.discr_sequence_space(shape, dtype='complex128')
    ran_hc = odl.discr_sequence_space(shape, dtype='float64')
    dom = odl.discr_sequence_space(shape, dtype='complex128')
    dom_hc = odl.discr_sequence_space((3, 5), dtype='complex128')

    # Implicit range
    DiscreteFourierTransformInverse(dom, impl=impl)

    # Explicit range
    DiscreteFourierTransformInverse(ran, domain=dom, impl=impl)
    DiscreteFourierTransformInverse(ran_hc, domain=dom_hc, axes=(0,),
                                    impl=impl, halfcomplex=True)


def test_dft_call(impl):

    # 2d, complex, all ones and random back & forth
    shape = (4, 5)
    dft_dom = odl.discr_sequence_space(shape, dtype='complex64')
    dft = DiscreteFourierTransform(domain=dft_dom, impl=impl)
    idft = DiscreteFourierTransformInverse(range=dft_dom, impl=impl)

    assert dft.domain == idft.range
    assert dft.range == idft.domain

    one = dft.domain.one()
    one_dft1 = dft(one, flags=('FFTW_ESTIMATE',))
    one_dft2 = dft.inverse.inverse(one, flags=('FFTW_ESTIMATE',))
    one_dft3 = dft.adjoint.adjoint(one, flags=('FFTW_ESTIMATE',))
    true_dft = [[20, 0, 0, 0, 0],  # along all axes by default
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
    assert np.allclose(one_dft1, true_dft)
    assert np.allclose(one_dft2, true_dft)
    assert np.allclose(one_dft3, true_dft)

    one_idft1 = idft(one_dft1, flags=('FFTW_ESTIMATE',))
    one_idft2 = dft.inverse(one_dft1, flags=('FFTW_ESTIMATE',))
    one_idft3 = dft.adjoint(one_dft1, flags=('FFTW_ESTIMATE',))
    assert np.allclose(one_idft1, one)
    assert np.allclose(one_idft2, one)
    assert np.allclose(one_idft3, one)

    rand_arr = _random_array(shape, 'complex128')
    rand_arr_dft = dft(rand_arr, flags=('FFTW_ESTIMATE',))
    rand_arr_idft = idft(rand_arr_dft, flags=('FFTW_ESTIMATE',))
    assert (rand_arr_idft - rand_arr).norm() < 1e-6

    # 2d, halfcomplex, first axis
    shape = (4, 5)
    axes = 0
    dft_dom = odl.discr_sequence_space(shape, dtype='float32')
    dft = DiscreteFourierTransform(domain=dft_dom, impl=impl, halfcomplex=True,
                                   axes=axes)
    idft = DiscreteFourierTransformInverse(range=dft_dom, impl=impl,
                                           halfcomplex=True, axes=axes)

    assert dft.domain == idft.range
    assert dft.range == idft.domain

    one = dft.domain.one()
    one_dft = dft(one, flags=('FFTW_ESTIMATE',))
    true_dft = [[4, 4, 4, 4, 4],  # transform axis shortened
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
    assert np.allclose(one_dft, true_dft)

    one_idft1 = idft(one_dft, flags=('FFTW_ESTIMATE',))
    one_idft2 = dft.inverse(one_dft, flags=('FFTW_ESTIMATE',))
    assert np.allclose(one_idft1, one)
    assert np.allclose(one_idft2, one)

    rand_arr = _random_array(shape, 'complex128')
    rand_arr_dft = dft(rand_arr, flags=('FFTW_ESTIMATE',))
    rand_arr_idft = idft(rand_arr_dft, flags=('FFTW_ESTIMATE',))
    assert (rand_arr_idft - rand_arr).norm() < 1e-6


def test_dft_sign(impl):
    # Test if the FT sign behaves as expected, i.e. that the FT with sign
    # '+' and '-' have same real parts and opposite imaginary parts.

    # 2d, complex, all ones and random back & forth
    shape = (4, 5)
    dft_dom = odl.discr_sequence_space(shape, dtype='complex64')
    dft_minus = DiscreteFourierTransform(domain=dft_dom, impl=impl, sign='-')
    dft_plus = DiscreteFourierTransform(domain=dft_dom, impl=impl, sign='+')

    arr = dft_dom.element([[0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0]])
    arr_dft_minus = dft_minus(arr, flags=('FFTW_ESTIMATE',))
    arr_dft_plus = dft_plus(arr, flags=('FFTW_ESTIMATE',))

    assert all_almost_equal(arr_dft_minus.real, arr_dft_plus.real)
    assert all_almost_equal(arr_dft_minus.imag, -arr_dft_plus.imag)
    assert all_almost_equal(dft_minus.inverse(arr_dft_minus), arr)
    assert all_almost_equal(dft_plus.inverse(arr_dft_plus), arr)
    assert all_almost_equal(dft_minus.inverse.inverse(arr), dft_minus(arr))
    assert all_almost_equal(dft_plus.inverse.inverse(arr), dft_plus(arr))

    # 2d, halfcomplex, first axis
    shape = (4, 5)
    axes = (0,)
    dft_dom = odl.discr_sequence_space(shape, dtype='float32')
    arr = dft_dom.element([[0, 0, 0, 0, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0]])

    dft = DiscreteFourierTransform(
        domain=dft_dom, impl=impl, halfcomplex=True, sign='-', axes=axes)

    arr_dft_minus = dft(arr, flags=('FFTW_ESTIMATE',))
    arr_idft_minus = dft.inverse(arr_dft_minus, flags=('FFTW_ESTIMATE',))

    assert all_almost_equal(arr_idft_minus, arr)

    with pytest.raises(ValueError):
        DiscreteFourierTransform(
            domain=dft_dom, impl=impl, halfcomplex=True, sign='+', axes=axes)


def test_dft_init_plan(impl):

    # 2d, halfcomplex, first axis
    shape = (4, 5)
    axes = 0
    dft_dom = odl.discr_sequence_space(shape, dtype='float32')

    dft = DiscreteFourierTransform(dft_dom, impl=impl, axes=axes,
                                   halfcomplex=True)

    if impl != 'pyfftw':
        with pytest.raises(ValueError):
            dft.init_fftw_plan()
        with pytest.raises(ValueError):
            dft.clear_fftw_plan()
    else:
        dft.init_fftw_plan()

        # Make sure plan can be used
        dft._fftw_plan(dft.domain.element().asarray(),
                       dft.range.element().asarray())
        dft.clear_fftw_plan()
        assert dft._fftw_plan is None


# ---- FourierTransform ---- #


def test_fourier_trafo_range(exponent, dtype):
    # Check if the range is initialized correctly. Encompasses the init test

    # Testing R2C for real dtype, else C2C

    # 1D
    shape = 10
    space_discr = odl.uniform_discr(0, 1, shape, exponent=exponent,
                                    impl='numpy', dtype=dtype)

    dft = FourierTransform(space_discr, halfcomplex=True, shift=True)
    assert dft.range.field == odl.ComplexNumbers()
    halfcomplex = True if is_real_dtype(dtype) else False
    assert dft.range.grid == reciprocal(dft.domain.grid,
                                        halfcomplex=halfcomplex,
                                        shift=True)
    assert dft.range.exponent == conj_exponent(exponent)

    # 3D
    shape = (3, 4, 5)
    space_discr = odl.uniform_discr([0] * 3, [1] * 3, shape, exponent=exponent,
                                    impl='numpy', dtype=dtype)

    dft = FourierTransform(space_discr, halfcomplex=True, shift=True)
    assert dft.range.field == odl.ComplexNumbers()
    halfcomplex = True if is_real_dtype(dtype) else False
    assert dft.range.grid == reciprocal(dft.domain.grid,
                                        halfcomplex=halfcomplex,
                                        shift=True)
    assert dft.range.exponent == conj_exponent(exponent)

    # shift must be True in the last axis
    if halfcomplex:
        with pytest.raises(ValueError):
            FourierTransform(space_discr, shift=(True, True, False))

    if exponent != 2.0:
        with pytest.raises(NotImplementedError):
            dft.adjoint

    with pytest.raises(TypeError):
        FourierTransform(dft.domain.partition)


def test_fourier_trafo_init_plan(impl, dtype):

    # Not supported, skip
    if dtype == np.dtype('float16') and impl == 'pyfftw':
        return

    shape = 10
    halfcomplex, _ = _params_from_dtype(dtype)

    space_discr = odl.uniform_discr(0, 1, shape, dtype=dtype)

    ft = FourierTransform(space_discr, impl=impl, halfcomplex=halfcomplex)
    if impl != 'pyfftw':
        with pytest.raises(ValueError):
            ft.init_fftw_plan()
        with pytest.raises(ValueError):
            ft.clear_fftw_plan()
    else:
        ft.init_fftw_plan()

        # Make sure plan can be used
        ft._fftw_plan(ft.domain.element().asarray(),
                      ft.range.element().asarray())
        ft.clear_fftw_plan()
        assert ft._fftw_plan is None

    # With temporaries
    ft.create_temporaries(r=True, f=False)
    if impl != 'pyfftw':
        with pytest.raises(ValueError):
            ft.init_fftw_plan()
        with pytest.raises(ValueError):
            ft.clear_fftw_plan()
    else:
        ft.init_fftw_plan()

        # Make sure plan can be used
        ft._fftw_plan(ft.domain.element().asarray(),
                      ft.range.element().asarray())
        ft.clear_fftw_plan()
        assert ft._fftw_plan is None

    ft.create_temporaries(r=False, f=True)
    if impl != 'pyfftw':
        with pytest.raises(ValueError):
            ft.init_fftw_plan()
        with pytest.raises(ValueError):
            ft.clear_fftw_plan()
    else:
        ft.init_fftw_plan()

        # Make sure plan can be used
        ft._fftw_plan(ft.domain.element().asarray(),
                      ft.range.element().asarray())
        ft.clear_fftw_plan()
        assert ft._fftw_plan is None


def test_fourier_trafo_create_temp():

    shape = 10
    space_discr = odl.uniform_discr(0, 1, shape, dtype='complex64')

    ft = FourierTransform(space_discr)
    ft.create_temporaries()
    assert ft._tmp_r is not None
    assert ft._tmp_f is not None

    ift = ft.inverse
    assert ift._tmp_r is not None
    assert ift._tmp_f is not None

    ft.clear_temporaries()
    assert ft._tmp_r is None
    assert ft._tmp_f is None


def test_fourier_trafo_call(impl, dtype):
    # Test if all variants can be called without error

    # Not supported, skip
    if dtype == np.dtype('float16') and impl == 'pyfftw':
        return

    shape = 10
    halfcomplex, _ = _params_from_dtype(dtype)
    space_discr = odl.uniform_discr(0, 1, shape, dtype=dtype)

    ft = FourierTransform(space_discr, impl=impl, halfcomplex=halfcomplex)
    ift = ft.inverse

    one = space_discr.one()
    assert np.allclose(ift(ft(one)), one)

    # With temporaries
    ft.create_temporaries()
    ift = ft.inverse  # shares temporaries
    one = space_discr.one()
    assert np.allclose(ift(ft(one)), one)


def sinc(x):
    # numpy.sinc scales by pi, we don't want that
    return np.sinc(x / np.pi)


def test_fourier_trafo_charfun_1d():
    # Characteristic function of [0, 1], its Fourier transform is
    # given by exp(-1j * y / 2) * sinc(y/2)
    def char_interval(x):
        return (x >= 0) & (x <= 1)

    def char_interval_ft(x):
        return np.exp(-1j * x / 2) * sinc(x / 2) / np.sqrt(2 * np.pi)

    # Base version
    discr = odl.uniform_discr(-2, 2, 40, impl='numpy')
    dft_base = FourierTransform(discr)

    # Complex version, should be as good
    discr = odl.uniform_discr(-2, 2, 40, impl='numpy', dtype='complex64')
    dft_complex = FourierTransform(discr)

    # Without shift
    discr = odl.uniform_discr(-2, 2, 40, impl='numpy', dtype='complex64')
    dft_complex_shift = FourierTransform(discr, shift=False)

    for dft in [dft_base, dft_complex, dft_complex_shift]:
        func_true_ft = dft.range.element(char_interval_ft)
        func_dft = dft(char_interval)
        assert (func_dft - func_true_ft).norm() < 1e-6


def test_fourier_trafo_scaling():
    # Test if the FT scales correctly

    # Characteristic function of [0, 1], its Fourier transform is
    # given by exp(-1j * y / 2) * sinc(y/2)
    def char_interval(x):
        return (x >= 0) & (x <= 1)

    def char_interval_ft(x):
        return np.exp(-1j * x / 2) * sinc(x / 2) / np.sqrt(2 * np.pi)

    fspace = odl.FunctionSpace(odl.IntervalProd(-2, 2),
                               field=odl.ComplexNumbers())
    discr = odl.uniform_discr_fromspace(fspace, 40, impl='numpy')
    dft = FourierTransform(discr)

    for factor in (2, 1j, -2.5j, 1 - 4j):
        func_true_ft = factor * dft.range.element(char_interval_ft)
        func_dft = dft(factor * fspace.element(char_interval))
        assert (func_dft - func_true_ft).norm() < 1e-6


def test_fourier_trafo_sign(impl):
    # Test if the FT sign behaves as expected, i.e. that the FT with sign
    # '+' and '-' have same real parts and opposite imaginary parts.

    def char_interval(x):
        return (x >= 0) & (x <= 1)

    discr = odl.uniform_discr(-2, 2, 40, impl='numpy', dtype='complex64')
    ft_minus = FourierTransform(discr, sign='-', impl=impl)
    ft_plus = FourierTransform(discr, sign='+', impl=impl)

    func_ft_minus = ft_minus(char_interval)
    func_ft_plus = ft_plus(char_interval)
    assert np.allclose(func_ft_minus.real, func_ft_plus.real)
    assert np.allclose(func_ft_minus.imag, -func_ft_plus.imag)
    assert np.allclose(ft_minus.inverse.inverse(char_interval),
                       ft_minus(char_interval))
    assert np.allclose(ft_plus.inverse.inverse(char_interval),
                       ft_plus(char_interval))

    discr = odl.uniform_discr(-2, 2, 40, impl='numpy', dtype='float32')
    with pytest.raises(ValueError):
        FourierTransform(discr, sign='+', impl=impl, halfcomplex=True)
    with pytest.raises(ValueError):
        FourierTransform(discr, sign=-1, impl=impl)


def test_fourier_trafo_inverse(impl, sign):
    # Test if the inverse really is the inverse

    def char_interval(x):
        return (x >= 0) & (x <= 1)

    # Complex-to-complex
    discr = odl.uniform_discr(-2, 2, 40, impl='numpy', dtype='complex64')
    discr_char = discr.element(char_interval)

    ft = FourierTransform(discr, sign=sign, impl=impl)
    assert all_almost_equal(ft.inverse(ft(char_interval)), discr_char)
    assert all_almost_equal(ft.adjoint(ft(char_interval)), discr_char)

    # Half-complex
    discr = odl.uniform_discr(-2, 2, 40, impl='numpy', dtype='float32')
    ft = FourierTransform(discr, impl=impl, halfcomplex=True)
    assert all_almost_equal(ft.inverse(ft(char_interval)), discr_char)

    def char_rect(x):
        return (x[0] >= 0) & (x[0] <= 1) & (x[1] >= 0) & (x[1] <= 1)

    # 2D with axes, C2C
    discr = odl.uniform_discr([-2, -2], [2, 2], (20, 10), impl='numpy',
                              dtype='complex64')
    discr_rect = discr.element(char_rect)

    for axes in [(0,), 1]:
        ft = FourierTransform(discr, sign=sign, impl=impl, axes=axes)
        assert all_almost_equal(ft.inverse(ft(char_rect)), discr_rect)
        assert all_almost_equal(ft.adjoint(ft(char_rect)), discr_rect)

    # 2D with axes, halfcomplex
    discr = odl.uniform_discr([-2, -2], [2, 2], (20, 10), impl='numpy',
                              dtype='float32')
    discr_rect = discr.element(char_rect)

    for halfcomplex in [False, True]:
        if halfcomplex and sign == '+':
            continue  # cannot mix halfcomplex with sign

        for axes in [(0,), (1,)]:
            ft = FourierTransform(discr, sign=sign, impl=impl, axes=axes,
                                  halfcomplex=halfcomplex)
            assert all_almost_equal(ft.inverse(ft(char_rect)), discr_rect)
            assert all_almost_equal(ft.adjoint(ft(char_rect)), discr_rect)


def test_fourier_trafo_hat_1d():
    # Hat function as used in linear interpolation. It is not so
    # well discretized by nearest neighbor interpolation, so a larger
    # error is to be expected.
    def hat_func(x):
        out = np.where(x < 0, 1 + x, 1 - x)
        out[x < -1] = 0
        out[x > 1] = 0
        return out

    def hat_func_ft(x):
        return sinc(x / 2) ** 2 / np.sqrt(2 * np.pi)

    # Using a single-precision implementation, should be as good
    # With linear interpolation in the discretization, should be better?
    for interp in ['nearest', 'linear']:
        discr = odl.uniform_discr(-2, 2, 101, impl='numpy', dtype='float32',
                                  interp=interp)
        dft = FourierTransform(discr)
        func_true_ft = dft.range.element(hat_func_ft)
        func_dft = dft(hat_func)
        assert (func_dft - func_true_ft).norm() < 0.001


def test_fourier_trafo_complex_sum():
    # Sum of characteristic function and hat function, both with
    # known FT's.
    def hat_func(x):
        out = 1 - np.abs(x)
        out[x < -1] = 0
        out[x > 1] = 0
        return out

    def hat_func_ft(x):
        return sinc(x / 2) ** 2 / np.sqrt(2 * np.pi)

    def char_interval(x):
        return (x >= 0) & (x <= 1)

    def char_interval_ft(x):
        return np.exp(-1j * x / 2) * sinc(x / 2) / np.sqrt(2 * np.pi)

    discr = odl.uniform_discr(-2, 2, 200, impl='numpy', dtype='complex128')
    dft = FourierTransform(discr, shift=False)

    func = discr.element(hat_func) + 1j * discr.element(char_interval)
    func_true_ft = (dft.range.element(hat_func_ft) +
                    1j * dft.range.element(char_interval_ft))
    func_dft = dft(func)
    assert (func_dft - func_true_ft).norm() < 0.001


def test_fourier_trafo_gaussian_1d():
    # Gaussian function, will be mapped to itself. Truncation error is
    # relatively large, though, we need a large support.
    def gaussian(x):
        return np.exp(-x ** 2 / 2)

    discr = odl.uniform_discr(-10, 10, 201, impl='numpy')
    dft = FourierTransform(discr)
    func_true_ft = dft.range.element(gaussian)
    func_dft = dft(gaussian)
    assert (func_dft - func_true_ft).norm() < 0.001


def test_fourier_trafo_freq_shifted_charfun_1d():
    # Frequency-shifted characteristic function: mult. with
    # exp(-1j * b * x) corresponds to shifting the FT by b.
    def fshift_char_interval(x):
        return np.exp(-1j * x * np.pi) * ((x >= -0.5) & (x <= 0.5))

    def fshift_char_interval_ft(x):
        return sinc((x + np.pi) / 2) / np.sqrt(2 * np.pi)

    # Number of points is very important here (aliasing)
    discr = odl.uniform_discr(-2, 2, 400, impl='numpy',
                              dtype='complex64')
    dft = FourierTransform(discr)
    func_true_ft = dft.range.element(fshift_char_interval_ft)
    func_dft = dft(fshift_char_interval)
    assert (func_dft - func_true_ft).norm() < 0.001


def test_dft_with_known_pairs_2d():

    # Frequency-shifted product of characteristic functions
    def fshift_char_rect(x):
        # Characteristic function of the cuboid
        # [-1, 1] x [1, 2]
        return (x[0] >= -1) & (x[0] <= 1) & (x[1] >= 1) & (x[1] <= 2)

    def fshift_char_rect_ft(x):
        # FT is a product of shifted and frequency-shifted sinc functions
        # 1st comp.: 2 * sinc(y)
        # 2nd comp.: exp(-1j * y * 3/2) * sinc(y/2)
        # Overall factor: (2 * pi)^(-1)
        return (2 * sinc(x[0]) *
                np.exp(-1j * x[1] * 3 / 2) * sinc(x[1] / 2) /
                (2 * np.pi))

    discr = odl.uniform_discr([-2] * 2, [2] * 2, (100, 400), impl='numpy',
                              dtype='complex64')
    dft = FourierTransform(discr)
    func_true_ft = dft.range.element(fshift_char_rect_ft)
    func_dft = dft(fshift_char_rect)
    assert (func_dft - func_true_ft).norm() < 0.001


def test_fourier_trafo_completely():
    # Complete explicit test of all FT components on two small examples

    # Discretization with 4 points
    discr = odl.uniform_discr(-2, 2, 4, dtype='complex')
    # Interval boundaries -2, -1, 0, 1, 2
    assert np.allclose(discr.partition.cell_boundary_vecs[0],
                       [-2, -1, 0, 1, 2])
    # Grid points -1.5, -0.5, 0.5, 1.5
    assert np.allclose(discr.grid.coord_vectors[0],
                       [-1.5, -0.5, 0.5, 1.5])

    # First test function, symmetric. Can be represented exactly in the
    # discretization.
    def f(x):
        return (x >= -1) & (x <= 1)

    def fhat(x):
        return np.sqrt(2 / np.pi) * sinc(x)

    # Discretize f, check values
    f_discr = discr.element(f)
    assert np.allclose(f_discr, [0, 1, 1, 0])

    # "s" = shifted, "n" = not shifted

    # Reciprocal grids
    recip_s = reciprocal(discr.grid, shift=True)
    recip_n = reciprocal(discr.grid, shift=False)
    assert np.allclose(recip_s.coord_vectors[0],
                       np.linspace(-np.pi, np.pi / 2, 4))
    assert np.allclose(recip_n.coord_vectors[0],
                       np.linspace(-3 * np.pi / 4, 3 * np.pi / 4, 4))

    # Range
    range_part_s = odl.uniform_partition_fromgrid(recip_s)
    range_s = odl.uniform_discr_frompartition(range_part_s, dtype='complex')
    range_part_n = odl.uniform_partition_fromgrid(recip_n)
    range_n = odl.uniform_discr_frompartition(range_part_n, dtype='complex')

    # Pre-processing
    preproc_s = [1, -1, 1, -1]
    preproc_n = [np.exp(1j * 3 / 4 * np.pi * k) for k in range(4)]

    fpre_s = dft_preprocess_data(f_discr, shift=True)
    fpre_n = dft_preprocess_data(f_discr, shift=False)
    assert np.allclose(fpre_s, f_discr * discr.element(preproc_s))
    assert np.allclose(fpre_n, f_discr * discr.element(preproc_n))

    # FFT step, replicating the _call_numpy method
    fft_s = np.fft.fftn(fpre_s, s=discr.shape, axes=[0])
    fft_n = np.fft.fftn(fpre_n, s=discr.shape, axes=[0])
    assert np.allclose(fft_s, [0, -1 + 1j, 2, -1 - 1j])
    assert np.allclose(
        fft_n, [np.exp(1j * np.pi * (3 - 2 * k) / 4) +
                np.exp(1j * np.pi * (3 - 2 * k) / 2)
                for k in range(4)])

    # Interpolation kernel FT
    interp_s = np.sinc(np.linspace(-1 / 2, 1 / 4, 4)) / np.sqrt(2 * np.pi)
    interp_n = np.sinc(np.linspace(-3 / 8, 3 / 8, 4)) / np.sqrt(2 * np.pi)
    assert np.allclose(interp_s,
                       _interp_kernel_ft(np.linspace(-1 / 2, 1 / 4, 4),
                                         interp='nearest'))
    assert np.allclose(interp_n,
                       _interp_kernel_ft(np.linspace(-3 / 8, 3 / 8, 4),
                                         interp='nearest'))

    # Post-processing
    postproc_s = np.exp(1j * np.pi * np.linspace(-3 / 2, 3 / 4, 4))
    postproc_n = np.exp(1j * np.pi * np.linspace(-9 / 8, 9 / 8, 4))

    fpost_s = dft_postprocess_data(
        range_s.element(fft_s), real_grid=discr.grid, recip_grid=recip_s,
        shift=[True], axes=(0,), interp='nearest')
    fpost_n = dft_postprocess_data(
        range_n.element(fft_n), real_grid=discr.grid, recip_grid=recip_n,
        shift=[False], axes=(0,), interp='nearest')

    assert np.allclose(fpost_s, fft_s * postproc_s * interp_s)
    assert np.allclose(fpost_n, fft_n * postproc_n * interp_n)

    # Comparing to the known result sqrt(2/pi) * sinc(x)
    assert np.allclose(fpost_s, fhat(recip_s.coord_vectors[0]))
    assert np.allclose(fpost_n, fhat(recip_n.coord_vectors[0]))

    # Doing the exact same with direct application of the FT operator
    ft_op_s = FourierTransform(discr, shift=True)
    ft_op_n = FourierTransform(discr, shift=False)
    assert ft_op_s.range.grid == recip_s
    assert ft_op_n.range.grid == recip_n

    ft_f_s = ft_op_s(f)
    ft_f_n = ft_op_n(f)
    assert np.allclose(ft_f_s, fhat(recip_s.coord_vectors[0]))
    assert np.allclose(ft_f_n, fhat(recip_n.coord_vectors[0]))

    # Second test function, asymmetric. Can also be represented exactly in the
    # discretization.
    def f(x):
        return (x >= 0) & (x <= 1)

    def fhat(x):
        return np.exp(-1j * x / 2) * sinc(x / 2) / np.sqrt(2 * np.pi)

    # Discretize f, check values
    f_discr = discr.element(f)
    assert np.allclose(f_discr, [0, 0, 1, 0])

    # Pre-processing
    fpre_s = dft_preprocess_data(f_discr, shift=True)
    fpre_n = dft_preprocess_data(f_discr, shift=False)
    assert np.allclose(fpre_s, [0, 0, 1, 0])
    assert np.allclose(fpre_n, [0, 0, -1j, 0])

    # FFT step
    fft_s = np.fft.fftn(fpre_s, s=discr.shape, axes=[0])
    fft_n = np.fft.fftn(fpre_n, s=discr.shape, axes=[0])
    assert np.allclose(fft_s, [1, -1, 1, -1])
    assert np.allclose(fft_n, [-1j, 1j, -1j, 1j])

    fpost_s = dft_postprocess_data(
        range_s.element(fft_s), real_grid=discr.grid, recip_grid=recip_s,
        shift=[True], axes=(0,), interp='nearest')
    fpost_n = dft_postprocess_data(
        range_n.element(fft_n), real_grid=discr.grid, recip_grid=recip_n,
        shift=[False], axes=(0,), interp='nearest')

    assert np.allclose(fpost_s, fft_s * postproc_s * interp_s)
    assert np.allclose(fpost_n, fft_n * postproc_n * interp_n)

    # Comparing to the known result exp(-1j*x/2) * sinc(x/2) / sqrt(2*pi)
    assert np.allclose(fpost_s, fhat(recip_s.coord_vectors[0]))
    assert np.allclose(fpost_n, fhat(recip_n.coord_vectors[0]))

    # Doing the exact same with direct application of the FT operator
    ft_f_s = ft_op_s(f)
    ft_f_n = ft_op_n(f)
    assert np.allclose(ft_f_s, fhat(recip_s.coord_vectors[0]))
    assert np.allclose(ft_f_n, fhat(recip_n.coord_vectors[0]))

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
