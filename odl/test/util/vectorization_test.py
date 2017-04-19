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

from odl.discr.grid import sparse_meshgrid
from odl.util import is_int_dtype
from odl.util.testutils import all_equal
from odl.util.vectorization import (
    is_valid_input_array, is_valid_input_meshgrid,
    out_shape_from_meshgrid, out_shape_from_array,
    vectorize)


def test_is_valid_input_array():

    # 1d
    valid_shapes = [(1, 1), (1, 2), (1, 20), (20,)]
    invalid_shapes = [(2, 1), (1, 1, 1), (1,), ()]

    for shp in valid_shapes:
        arr = np.zeros(shp)
        assert is_valid_input_array(arr, ndim=1)

    for shp in invalid_shapes:
        arr = np.zeros(shp)
        assert not is_valid_input_array(arr, ndim=1)

    # 3d
    valid_shapes = [(3, 1), (3, 2), (3, 20)]
    invalid_shapes = [(3,), (20,), (4, 1), (3, 1, 1), ()]

    for shp in valid_shapes:
        arr = np.zeros(shp)
        assert is_valid_input_array(arr, ndim=3)

    for shp in invalid_shapes:
        arr = np.zeros(shp)
        assert not is_valid_input_array(arr, ndim=3)

    # Other input
    assert is_valid_input_array([[1, 2], [3, 4]], ndim=2)
    invalid_input = [1, [[[1, 2], [3, 4]]], (5,)]
    for inp in invalid_input:
        assert not is_valid_input_array(inp, ndim=2)


def test_is_valid_input_meshgrid():

    # 1d
    x = np.zeros(2)

    valid_mg = sparse_meshgrid(x)
    assert is_valid_input_meshgrid(valid_mg, ndim=1)

    invalid_mg = sparse_meshgrid(x, x)
    assert not is_valid_input_meshgrid(invalid_mg, ndim=1)

    x = np.zeros((2, 2))
    invalid_mg = sparse_meshgrid(x)
    assert not is_valid_input_meshgrid(invalid_mg, ndim=1)

    # 3d
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)

    valid_mg = sparse_meshgrid(x, y, z)
    assert is_valid_input_meshgrid(valid_mg, ndim=3)

    invalid_mg = sparse_meshgrid(x, x, y, z)
    assert not is_valid_input_meshgrid(invalid_mg, ndim=3)

    x = np.zeros((3, 3))
    invalid_mg = sparse_meshgrid(x)
    assert not is_valid_input_meshgrid(invalid_mg, ndim=3)

    # Other input
    invalid_input = [1, [1, 2], ([1, 2], [3, 4]), (5,), np.zeros((2, 2))]
    for inp in invalid_input:
        assert not is_valid_input_meshgrid(inp, ndim=1)
        assert not is_valid_input_meshgrid(inp, ndim=2)


def test_out_shape_from_array():

    # 1d
    arr = np.zeros((1, 1))
    assert out_shape_from_array(arr) == (1,)
    arr = np.zeros((1, 2))
    assert out_shape_from_array(arr) == (2,)
    arr = np.zeros((1,))
    assert out_shape_from_array(arr) == (1,)
    arr = np.zeros((20,))
    assert out_shape_from_array(arr) == (20,)

    # 3d
    arr = np.zeros((3, 1))
    assert out_shape_from_array(arr) == (1,)
    arr = np.zeros((3, 2))
    assert out_shape_from_array(arr) == (2,)
    arr = np.zeros((3, 20))
    assert out_shape_from_array(arr) == (20,)


def test_out_shape_from_meshgrid():

    # 1d
    x = np.zeros(2)
    mg = sparse_meshgrid(x)
    assert out_shape_from_meshgrid(mg) == (2,)

    # 3d
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)
    mg = sparse_meshgrid(x, y, z)
    assert out_shape_from_meshgrid(mg) == (2, 3, 4)

    # 3d, fleshed out meshgrids
    x, y, z = np.zeros(2), np.zeros(3), np.zeros(4)
    mg = np.meshgrid(x, y, z, sparse=False, indexing='ij', copy=True)
    assert out_shape_from_meshgrid(mg) == (2, 3, 4)

    mg = np.meshgrid(x, y, z, sparse=False, indexing='ij', copy=True)
    mg = tuple(reversed([np.asfortranarray(arr) for arr in mg]))
    assert out_shape_from_meshgrid(mg) == (2, 3, 4)


def test_vectorize_1d_otype():

    import sys

    # Test vectorization in 1d with given data type for output
    arr = (np.arange(5) - 2)[None, :]
    mg = sparse_meshgrid(np.arange(5) - 3)
    val_1 = -1
    val_2 = 2

    @vectorize(otypes=['int'])
    def simple_func(x):
        return 0 if x < 0 else 1

    true_result_arr = [0, 0, 1, 1, 1]
    true_result_mg = [0, 0, 0, 1, 1]

    # Out-of-place
    out = simple_func(arr)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.dtype('int')
    assert out.shape == (5,)
    assert all_equal(out, true_result_arr)

    out = simple_func(mg)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5,)
    assert out.dtype == np.dtype('int')
    assert all_equal(out, true_result_mg)

    assert simple_func(val_1) == 0
    assert simple_func(val_2) == 1

    # Python 2 really swallows this stuff in comparisons...
    bogus_input = [lambda x: x, object, Exception]
    if sys.version_info.major > 2:
        for b in bogus_input:
            with pytest.raises(TypeError):
                simple_func(b)

    # In-place
    out = np.empty(5, dtype='int')
    simple_func(arr, out=out)
    assert all_equal(out, true_result_arr)

    out = np.empty(5, dtype='int')
    simple_func(mg, out=out)
    assert all_equal(out, true_result_mg)


def test_vectorize_1d_lazy():

    # Test vectorization in 1d without data type --> lazy vectorization
    arr = (np.arange(5) - 2)[None, :]
    mg = sparse_meshgrid(np.arange(5) - 3)
    val_1 = -1
    val_2 = 2

    @vectorize
    def simple_func(x):
        return 0 if x < 0 else 1

    true_result_arr = [0, 0, 1, 1, 1]
    true_result_mg = [0, 0, 0, 1, 1]

    # Out-of-place
    out = simple_func(arr)
    assert isinstance(out, np.ndarray)
    assert is_int_dtype(out.dtype)
    assert out.shape == (5,)
    assert all_equal(out, true_result_arr)

    out = simple_func(mg)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5,)
    assert is_int_dtype(out.dtype)
    assert all_equal(out, true_result_mg)

    assert simple_func(val_1) == 0
    assert simple_func(val_2) == 1


def test_vectorize_2d_dtype():

    # Test vectorization in 2d with given data type for output
    arr = np.empty((2, 5), dtype='int')
    arr[0] = ([-3, -2, -1, 0, 1])
    arr[1] = ([-1, 0, 1, 2, 3])
    mg = sparse_meshgrid([-3, -2, -1, 0, 1], [-1, 0, 1, 2, 3])
    val_1 = (-1, 1)
    val_2 = (2, 1)

    @vectorize(otypes=['int'])
    def simple_func(x):
        return 0 if x[0] < 0 and x[1] > 0 else 1

    true_result_arr = [1, 1, 0, 1, 1]

    true_result_mg = [[1, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]]

    # Out-of-place
    out = simple_func(arr)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.dtype('int')
    assert out.shape == (5,)
    assert all_equal(out, true_result_arr)

    out = simple_func(mg)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.dtype('int')
    assert out.shape == (5, 5)
    assert all_equal(out, true_result_mg)

    assert simple_func(val_1) == 0
    assert simple_func(val_2) == 1

    # In-place
    out = np.empty(5, dtype='int')
    simple_func(arr, out=out)
    assert all_equal(out, true_result_arr)

    out = np.empty((5, 5), dtype='int')
    simple_func(mg, out=out)
    assert all_equal(out, true_result_mg)


def test_vectorize_2d_lazy():

    # Test vectorization in 1d without data type --> lazy vectorization
    arr = np.empty((2, 5), dtype='int')
    arr[0] = ([-3, -2, -1, 0, 1])
    arr[1] = ([-1, 0, 1, 2, 3])
    mg = sparse_meshgrid([-3, -2, -1, 0, 1], [-1, 0, 1, 2, 3])
    val_1 = (-1, 1)
    val_2 = (2, 1)

    @vectorize
    def simple_func(x):
        return 0 if x[0] < 0 and x[1] > 0 else 1

    true_result_arr = [1, 1, 0, 1, 1]

    true_result_mg = [[1, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 1, 0, 0, 0],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]]

    # Out-of-place
    out = simple_func(arr)
    assert isinstance(out, np.ndarray)
    assert is_int_dtype(out.dtype)
    assert out.shape == (5,)
    assert all_equal(out, true_result_arr)

    out = simple_func(mg)
    assert isinstance(out, np.ndarray)
    assert is_int_dtype(out.dtype)
    assert out.shape == (5, 5)
    assert all_equal(out, true_result_mg)

    assert simple_func(val_1) == 0
    assert simple_func(val_2) == 1


def test_vectorize_callable_class():

    # Test vectorization in 1d without data type --> lazy vectorization
    arr = [[-2, -1, 0, 1, 2]]
    mg = [[-3, -2, -1, 0, 1]]
    val_1 = -1
    val_2 = 2

    # Class with __call__ method
    class CallableClass(object):
        def __call__(self, x):
            return 0 if x < 0 else 1

    vectorized_call = vectorize(CallableClass())

    true_result_arr = [0, 0, 1, 1, 1]
    true_result_mg = [0, 0, 0, 1, 1]

    # Out-of-place
    out = vectorized_call(arr)
    assert isinstance(out, np.ndarray)
    assert is_int_dtype(out.dtype)
    assert out.shape == (5,)
    assert all_equal(out, true_result_arr)

    out = vectorized_call(mg)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5,)
    assert is_int_dtype(out.dtype)
    assert all_equal(out, true_result_mg)

    assert vectorized_call(val_1) == 0
    assert vectorized_call(val_2) == 1


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
