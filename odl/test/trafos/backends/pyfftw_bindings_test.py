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

import numpy as np
import pytest

from odl.trafos.backends import pyfftw_call, PYFFTW_AVAILABLE
from odl.util import (
    all_almost_equal, is_real_dtype, complex_dtype)


pytestmark = pytest.mark.skipif(not PYFFTW_AVAILABLE,
                                reason='`pyfftw` backend not available')


# --- pytest fixtures --- #


plan_params = ['estimate', 'measure', 'patient', 'exhaustive']
plan_ids = [" planning = '{}' ".format(p) for p in plan_params]


@pytest.fixture(scope="module", ids=plan_ids, params=plan_params)
def planning(request):
    return request.param


@pytest.fixture(scope='module', ids=[' forward ', ' backward '],
                params=['forward', 'backward'])
def direction(request):
    return request.param


# --- helper functions --- #


def _random_array(shape, dtype):
    if is_real_dtype(dtype):
        return np.random.rand(*shape).astype(dtype)
    else:
        return (np.random.rand(*shape).astype(dtype) +
                1j * np.random.rand(*shape).astype(dtype))


def _params_from_dtype(dtype):
    if is_real_dtype(dtype):
        halfcomplex = True
    else:
        halfcomplex = False
    return halfcomplex, complex_dtype(dtype)


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


# ---- pyfftw_call ---- #


def test_pyfftw_call_forward(floating_dtype):
    # Test against Numpy's FFT
    if floating_dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, out_dtype = _params_from_dtype(floating_dtype)

    for shape in [(10,), (3, 4, 5)]:
        arr = _random_array(shape, floating_dtype)

        if halfcomplex:
            true_dft = np.fft.rfftn(arr)
            dft_arr = np.empty(_halfcomplex_shape(shape), dtype=out_dtype)
        else:
            true_dft = np.fft.fftn(arr)
            dft_arr = np.empty(shape, dtype=out_dtype)

        pyfftw_call(arr, dft_arr, direction='forward',
                    halfcomplex=halfcomplex, preserve_input=False)

        assert all_almost_equal(dft_arr, true_dft)


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


def test_pyfftw_call_backward(floating_dtype):
    # Test against Numpy's IFFT, no normalization
    if floating_dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, in_dtype = _params_from_dtype(floating_dtype)

    for shape in [(10,), (3, 4, 5)]:
        # Scaling happens wrt output (large) shape
        idft_scaling = np.prod(shape)

        if halfcomplex:
            arr = _random_array(_halfcomplex_shape(shape), in_dtype)
            true_idft = np.fft.irfftn(arr, shape) * idft_scaling
        else:
            arr = _random_array(shape, in_dtype)
            true_idft = np.fft.ifftn(arr) * idft_scaling

        idft_arr = np.empty(shape, dtype=floating_dtype)
        pyfftw_call(arr, idft_arr, direction='backward',
                    halfcomplex=halfcomplex)

        assert all_almost_equal(idft_arr, true_idft)


def test_pyfftw_call_bad_input(direction):

    # Complex

    # Bad dtype
    dtype_in = np.dtype('complex128')
    arr_in = np.empty(3, dtype=dtype_in)
    bad_dtypes_out = np.sctypes['float'] + np.sctypes['complex']
    if dtype_in in bad_dtypes_out:
        # This one is correct, so we remove it
        bad_dtypes_out.remove(dtype_in)

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
        with pytest.raises(ValueError):
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


def test_pyfftw_call_forward_with_axes(floating_dtype):
    if floating_dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, out_dtype = _params_from_dtype(floating_dtype)
    shape = (3, 4, 5)

    test_axes = [(0, 1), [1], (-1,), (1, 0), (-1, -2, -3)]
    for axes in test_axes:
        arr = _random_array(shape, floating_dtype)
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


def test_pyfftw_call_backward_with_axes(floating_dtype):
    if floating_dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, in_dtype = _params_from_dtype(floating_dtype)
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

        idft_arr = np.empty(shape, dtype=floating_dtype)
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


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
