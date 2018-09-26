# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
import pytest

import odl
from odl.trafos.backends import pyfftw_call, PYFFTW_AVAILABLE
from odl.util import (
    is_real_dtype, complex_dtype)
from odl.util.testutils import (
    all_almost_equal, simple_fixture)


pytestmark = pytest.mark.skipif(not PYFFTW_AVAILABLE,
                                reason='`pyfftw` backend not available')


# --- pytest fixtures --- #


planning = simple_fixture('planning', ['estimate', 'measure', 'patient',
                                       'exhaustive'])
direction = simple_fixture('direction', ['forward', 'backward'])


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


def test_pyfftw_call_forward(odl_floating_dtype):
    # Test against Numpy's FFT
    dtype = odl_floating_dtype
    if dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, dtype_out = _params_from_dtype(dtype)
    for shape in [(10,), (3, 4, 5)]:
        arr = _random_array(shape, dtype)

        if halfcomplex:
            true_dft = np.fft.rfftn(arr)
            dft_arr = np.empty(_halfcomplex_shape(shape), dtype=dtype_out)
        else:
            true_dft = np.fft.fftn(arr)
            dft_arr = np.empty(shape, dtype=dtype_out)

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


def test_pyfftw_call_backward(odl_floating_dtype):
    # Test against Numpy's IFFT, no normalization
    dtype = odl_floating_dtype
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


def test_pyfftw_call_forward_with_axes(odl_floating_dtype):
    dtype = odl_floating_dtype
    if dtype == np.dtype('float16'):  # not supported, skipping
        return

    halfcomplex, dtype_out = _params_from_dtype(dtype)
    shape = (3, 4, 5)
    test_axes = [(0, 1), [1], (-1,), (1, 0), (-1, -2, -3)]
    for axes in test_axes:
        arr = _random_array(shape, dtype)
        if halfcomplex:
            true_dft = np.fft.rfftn(arr, axes=axes)
            dft_arr = np.empty(_halfcomplex_shape(shape, axes),
                               dtype=dtype_out)
        else:
            true_dft = np.fft.fftn(arr, axes=axes)
            dft_arr = np.empty(shape, dtype=dtype_out)

        pyfftw_call(arr, dft_arr, direction='forward', axes=axes,
                    halfcomplex=halfcomplex)

        assert all_almost_equal(dft_arr, true_dft)


def test_pyfftw_call_backward_with_axes(odl_floating_dtype):
    dtype = odl_floating_dtype
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
    odl.util.test_file(__file__)
