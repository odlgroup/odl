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

# External
import pytest
import numpy as np

# Internal
from odl.util import apply_on_boundary, fast_1d_tensor_mult, resize_array
from odl.util.testutils import all_equal


# --- apply_on_boundary --- #


def test_apply_on_boundary_default():

    # 1d
    arr = np.ones(5)
    result = apply_on_boundary(arr, lambda x: x * 2)
    assert all_equal(arr, [1, 1, 1, 1, 1])
    assert all_equal(result, [2, 1, 1, 1, 2])

    # 3d
    arr = np.ones((3, 4, 5))
    result = apply_on_boundary(arr, lambda x: x * 2)
    true_arr = 2 * np.ones((3, 4, 5))
    true_arr[1:-1, 1:-1, 1:-1] = 1
    assert all_equal(result, true_arr)


def test_apply_on_boundary_func_sequence_2d():

    arr = np.ones((3, 5))
    result = apply_on_boundary(arr, [lambda x: x * 2, lambda x: x * 3])
    assert all_equal(result, [[2, 2, 2, 2, 2],
                              [3, 1, 1, 1, 3],
                              [2, 2, 2, 2, 2]])


def test_apply_on_boundary_multiple_times_2d():

    arr = np.ones((3, 5))
    result = apply_on_boundary(arr, lambda x: x * 2, only_once=False)
    assert all_equal(result, [[4, 2, 2, 2, 4],
                              [2, 1, 1, 1, 2],
                              [4, 2, 2, 2, 4]])


def test_apply_on_boundary_which_boundaries():

    # 1d
    arr = np.ones(5)
    which = ((True, False),)
    result = apply_on_boundary(arr, lambda x: x * 2, which_boundaries=which)
    assert all_equal(result, [2, 1, 1, 1, 1])

    # 2d
    arr = np.ones((3, 5))
    which = ((True, False), True)
    result = apply_on_boundary(arr, lambda x: x * 2, which_boundaries=which)
    assert all_equal(result, [[2, 2, 2, 2, 2],
                              [2, 1, 1, 1, 2],
                              [2, 1, 1, 1, 2]])


def test_apply_on_boundary_which_boundaries_multiple_times_2d():

    # 2d
    arr = np.ones((3, 5))
    which = ((True, False), True)
    result = apply_on_boundary(arr, lambda x: x * 2, which_boundaries=which,
                               only_once=False)
    assert all_equal(result, [[4, 2, 2, 2, 4],
                              [2, 1, 1, 1, 2],
                              [2, 1, 1, 1, 2]])


def test_apply_on_boundary_axis_order_2d():

    arr = np.ones((3, 5))
    axis_order = (-1, -2)
    result = apply_on_boundary(arr, [lambda x: x * 3, lambda x: x * 2],
                               axis_order=axis_order)
    assert all_equal(result, [[3, 2, 2, 2, 3],
                              [3, 1, 1, 1, 3],
                              [3, 2, 2, 2, 3]])


# --- fast_1d_tensor_mult --- #


def test_fast_1d_tensor_mult():

    # Full multiplication
    def simple_mult_3(x, y, z):
        return x[:, None, None] * y[None, :, None] * z[None, None, :]

    shape = (2, 3, 4)
    x, y, z = (np.arange(size, dtype='float64') for size in shape)
    true_result = simple_mult_3(x, y, z)

    # Standard call
    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [x, y, z])
    assert all_equal(out, true_result)
    assert all_equal(test_arr, np.ones(shape))  # no changes to input

    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [x, y, z], axes=(0, 1, 2))
    assert all_equal(out, true_result)

    # In-place with both same and different array as input
    test_arr = np.ones(shape)
    fast_1d_tensor_mult(test_arr, [x, y, z], out=test_arr)
    assert all_equal(test_arr, true_result)

    test_arr = np.ones(shape)
    out = np.empty(shape)
    fast_1d_tensor_mult(test_arr, [x, y, z], out=out)
    assert all_equal(out, true_result)

    # Different orderings
    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [y, x, z], axes=(1, 0, 2))
    assert all_equal(out, true_result)

    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [x, z, y], axes=(0, 2, 1))
    assert all_equal(out, true_result)

    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [z, x, y], axes=(2, 0, 1))
    assert all_equal(out, true_result)

    # More arrays than dimensions also ok with explicit axes
    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [z, x, y, np.ones(3)],
                              axes=(2, 0, 1, 1))
    assert all_equal(out, true_result)

    # Squeezable or extendable arrays also possible
    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [x, y, z[None, :]])
    assert all_equal(out, true_result)

    shape = (1, 3, 4)
    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [2, y, z])
    assert all_equal(out, simple_mult_3(np.ones(1) * 2, y, z))

    # Reduced multiplication, axis 0 not contained
    def simple_mult_2(y, z, nx):
        return np.ones((nx, 1, 1)) * y[None, :, None] * z[None, None, :]

    shape = (2, 3, 4)
    x, y, z = (np.arange(size, dtype='float64') for size in shape)
    true_result = simple_mult_2(y, z, 2)

    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [y, z], axes=(1, 2))
    assert all_equal(out, true_result)

    test_arr = np.ones(shape)
    out = fast_1d_tensor_mult(test_arr, [z, y], axes=(2, 1))
    assert all_equal(out, true_result)


def test_fast_1d_tensor_mult_error():

    shape = (2, 3, 4)
    test_arr = np.ones(shape)
    x, y, z = (np.arange(size, dtype='float64') for size in shape)

    # No ndarray to operate on
    with pytest.raises(TypeError):
        fast_1d_tensor_mult([[0, 0], [0, 0]], [x, x])

    # No 1d arrays given
    with pytest.raises(ValueError):
        fast_1d_tensor_mult(test_arr, [])

    # Length or dimension mismatch
    with pytest.raises(ValueError):
        fast_1d_tensor_mult(test_arr, [x, y])

    with pytest.raises(ValueError):
        fast_1d_tensor_mult(test_arr, [x, y], (1, 2, 0))

    with pytest.raises(ValueError):
        fast_1d_tensor_mult(test_arr, [x, x, y, z])

    # Axes out of bounds
    with pytest.raises(ValueError):
        fast_1d_tensor_mult(test_arr, [x, y, z], (1, 2, 3))

    with pytest.raises(ValueError):
        fast_1d_tensor_mult(test_arr, [x, y, z], (-2, -3, -4))

    # Other than 1d arrays
    with pytest.raises(ValueError):
        fast_1d_tensor_mult(test_arr, [x, y, np.ones((4, 2))], (-2, -3, -4))


# --- resize_array --- #


def test_resize_array_corner_cases():
    # Test extreme cases of resizing that are still valid

    # Resize to 0 total size
    arr = np.arange(12).reshape((3, 4))
    squashed_arr = resize_array(arr, (3, 0))
    assert squashed_arr.size == 0

    squashed_arr = resize_array(arr, (0, 0))
    assert squashed_arr.size == 0

    # Resize with periodic padding, using all values from the original
    # array on both sides
    max_per_shape = (9, 12)
    res_arr = resize_array(arr, max_per_shape, pad_mode='periodic')
    assert np.array_equal(res_arr, np.tile(arr, (3, 3)))

    # Symmetric padding, maximum number is one less compared to periodic
    # padding since the boundary value is not repeated
    arr = np.arange(6).reshape((2, 3))
    max_sym_shape = (4, 7)
    res_arr = resize_array(arr, max_sym_shape, pad_mode='symmetric')
    true_arr = np.array(
        [[5, 4, 3, 4, 5, 4, 3],
         [2, 1, 0, 1, 2, 1, 0],
         [5, 4, 3, 4, 5, 4, 3],
         [2, 1, 0, 1, 2, 1, 0]])
    assert np.array_equal(res_arr, true_arr)


def test_resize_array_raise():

    arr_1d = np.arange(6)
    arr_2d = np.arange(6).reshape((2, 3))

    # Shape not a sequence
    with pytest.raises(TypeError):
        resize_array(arr_1d, 19)

    # out given, but not an ndarray
    with pytest.raises(TypeError):
        resize_array(arr_1d, (10,), out=[])

    # out has wrong shape
    with pytest.raises(ValueError):
        out = np.empty((4, 5))
        resize_array(arr_2d, (5, 5), out=out)

    # Input and output arrays differ in dimensionality
    with pytest.raises(ValueError):
        out = np.empty((4, 5))
        resize_array(arr_1d, (4, 5))
    with pytest.raises(ValueError):
        out = np.empty((4, 5))
        resize_array(arr_1d, (4, 5), out=out)

    # Both frac_left and num_left given
    with pytest.raises(ValueError):
        resize_array(arr_1d, (10,), frac_left=0.3, num_left=3)

    # frac_left invalid
    with pytest.raises(ValueError):
        resize_array(arr_1d, (10,), frac_left=-0.1)
    with pytest.raises(ValueError):
        resize_array(arr_1d, (10,), frac_left=1.1)

    # invalid pad mode
    with pytest.raises(ValueError):
        resize_array(arr_1d, (10,), pad_mode='my_madeup_mode')

    # padding constant cannot be cast to output data type
    with pytest.raises(ValueError):
        resize_array(arr_1d, (10,), pad_const=1.0)  # arr_1d has dtype int
    with pytest.raises(ValueError):
        arr_1d_float = arr_1d.astype(float)
        resize_array(arr_1d_float, (10,), pad_const=1.0j)

    # Too few entries for order 0 or 1 padding modes
    empty_arr = np.ones((3, 0))
    with pytest.raises(ValueError):
        resize_array(empty_arr, (3, 1), pad_mode='order0')

    small_arr = np.ones((3, 1))
    with pytest.raises(ValueError):
        resize_array(small_arr, (3, 3), pad_mode='order1')

    # Too large padding sizes for symmetric
    small_arr = np.ones((3, 1))
    with pytest.raises(ValueError):
        resize_array(small_arr, (3, 3), pad_mode='symmetric')
    with pytest.raises(ValueError):
        resize_array(arr_2d, (5, 3), pad_mode='symmetric')
    with pytest.raises(ValueError):
        resize_array(arr_2d, (4, 3), frac_left=0, pad_mode='symmetric')

    # Too large padding sizes for periodic
    small_arr = np.ones((3, 1))
    with pytest.raises(ValueError):
        resize_array(small_arr, (3, 4), pad_mode='periodic')
    with pytest.raises(ValueError):
        resize_array(arr_2d, (7, 3), pad_mode='periodic')
    with pytest.raises(ValueError):
        resize_array(arr_2d, (5, 3), frac_left=0, pad_mode='symmetric')

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
