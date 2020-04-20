# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import numpy as np
import odl
import pytest
from odl.util import is_real_dtype
from odl.util.numerics import (
    _SUPPORTED_RESIZE_PAD_MODES, apply_on_boundary, binning,
    fast_1d_tensor_mult, resize_array)
from odl.util.testutils import (
    all_almost_equal, all_equal, dtype_tol, simple_fixture)

# --- pytest fixtures --- #


paddings = list(_SUPPORTED_RESIZE_PAD_MODES)
paddings.remove('constant')
paddings.extend([('constant', 0), ('constant', 1)])
padding_ids = [" pad_mode='{}'-{} ".format(*p)
               if isinstance(p, tuple)
               else " pad_mode='{}' ".format(p)
               for p in paddings]


@pytest.fixture(scope="module", ids=padding_ids, params=paddings)
def padding(request):
    if isinstance(request.param, tuple):
        pad_mode, pad_const = request.param
    else:
        pad_mode = request.param
        pad_const = 0

    return pad_mode, pad_const


variant = simple_fixture('variant', ['extend', 'restrict', 'mixed'])


@pytest.fixture(scope="module")
def resize_setup(padding, variant):
    array_in = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

    pad_mode, pad_const = padding

    if variant == 'extend':
        newshp = (4, 7)
        offset = (1, 2)

        if pad_mode == 'constant' and pad_const == 0:
            true_out = [[0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 2, 3, 4, 0],
                        [0, 0, 5, 6, 7, 8, 0],
                        [0, 0, 9, 10, 11, 12, 0]]
        elif pad_mode == 'constant' and pad_const == 1:
            true_out = [[1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 2, 3, 4, 1],
                        [1, 1, 5, 6, 7, 8, 1],
                        [1, 1, 9, 10, 11, 12, 1]]
        elif pad_mode == 'periodic':
            true_out = [[11, 12, 9, 10, 11, 12, 9],
                        [3, 4, 1, 2, 3, 4, 1],
                        [7, 8, 5, 6, 7, 8, 5],
                        [11, 12, 9, 10, 11, 12, 9]]
        elif pad_mode == 'symmetric':
            true_out = [[7, 6, 5, 6, 7, 8, 7],
                        [3, 2, 1, 2, 3, 4, 3],
                        [7, 6, 5, 6, 7, 8, 7],
                        [11, 10, 9, 10, 11, 12, 11]]
        elif pad_mode == 'order0':
            true_out = [[1, 1, 1, 2, 3, 4, 4],
                        [1, 1, 1, 2, 3, 4, 4],
                        [5, 5, 5, 6, 7, 8, 8],
                        [9, 9, 9, 10, 11, 12, 12]]
        elif pad_mode == 'order1':
            true_out = [[-5, -4, -3, -2, -1, 0, 1],
                        [-1, 0, 1, 2, 3, 4, 5],
                        [3, 4, 5, 6, 7, 8, 9],
                        [7, 8, 9, 10, 11, 12, 13]]

    elif variant == 'restrict':
        newshp = (2, 1)
        offset = (1, 2)

        true_out = [[7],
                    [11]]

    elif variant == 'mixed':
        newshp = (2, 7)
        offset = (1, 2)

        if pad_mode == 'constant' and pad_const == 0:
            true_out = [[0, 0, 5, 6, 7, 8, 0],
                        [0, 0, 9, 10, 11, 12, 0]]
        elif pad_mode == 'constant' and pad_const == 1:
            true_out = [[1, 1, 5, 6, 7, 8, 1],
                        [1, 1, 9, 10, 11, 12, 1]]
        elif pad_mode == 'periodic':
            true_out = [[7, 8, 5, 6, 7, 8, 5],
                        [11, 12, 9, 10, 11, 12, 9]]
        elif pad_mode == 'symmetric':
            true_out = [[7, 6, 5, 6, 7, 8, 7],
                        [11, 10, 9, 10, 11, 12, 11]]
        elif pad_mode == 'order0':
            true_out = [[5, 5, 5, 6, 7, 8, 8],
                        [9, 9, 9, 10, 11, 12, 12]]
        elif pad_mode == 'order1':
            true_out = [[3, 4, 5, 6, 7, 8, 9],
                        [7, 8, 9, 10, 11, 12, 13]]
        else:
            raise ValueError('unknown param')

    else:
        raise ValueError('unknown variant')

    return pad_mode, pad_const, newshp, offset, array_in, true_out


bin_size = simple_fixture(
    'bin_size', [0, 1, 2, 3, (1, 2), (4, 2), (8, 1), (2, 0), (2, 2, 2)]
)
bin_reduction = simple_fixture(
    'bin_reduction', [np.sum, np.max, np.mean], fmt=' {name}={value.__name__} '
)
bin_dtype = simple_fixture("bin_dtype", ['float64', 'int64'])


@pytest.fixture(scope="module")
def binning_setup(bin_size, bin_reduction, bin_dtype):
    arr = np.array([
        [0, 1, -1, 2, 0, 6],
        [-4, 2, 5, 2, -2, -1],
        [0, 0, 1, 0, 1, -1],
        [1, -1, 1, -1, 1, -1],
    ], dtype=bin_dtype)

    if bin_reduction == np.sum:
        out_dtype = bin_dtype

        if bin_size == 1:
            true_out_arr = arr.astype(out_dtype)
        elif bin_size == 2:
            true_out_arr = np.array([
                [-1, 8, 3],
                [0, 1, 0],
            ], dtype=out_dtype)
        elif bin_size == (1, 2):
            true_out_arr = np.array([
                [1, 1, 6],
                [-2, 7, -3],
                [0, 1, 0],
                [0, 0, 0],
            ], dtype=out_dtype)
        elif bin_size == (4, 2):
            true_out_arr = np.array([[-1, 9, 3]], dtype=out_dtype)
        else:
            true_out_arr = None

    elif bin_reduction == np.max:
        out_dtype = bin_dtype

        if bin_size == 1:
            true_out_arr = arr.astype(out_dtype)
        elif bin_size == 2:
            true_out_arr = np.array([
                [2, 5, 6],
                [1, 1, 1],
            ], dtype=out_dtype)
        elif bin_size == (1, 2):
            true_out_arr = np.array([
                [1, 2, 6],
                [2, 5, -1],
                [0, 1, 1],
                [1, 1, 1],
            ], dtype=out_dtype)
        elif bin_size == (4, 2):
            true_out_arr = np.array([[2, 5, 6]], dtype=out_dtype)
        else:
            true_out_arr = None

    elif bin_reduction == np.mean:
        out_dtype = np.result_type(bin_dtype, 1.0)

        if bin_size == 1:
            true_out_arr = arr.astype(out_dtype)
        elif bin_size == 2:
            true_out_arr = np.array([
                [-0.25, 2, 0.75],
                [0, 0.25, 0],
            ], dtype=out_dtype)
        elif bin_size == (1, 2):
            true_out_arr = np.array([
                [0.5, 0.5, 3],
                [-1, 3.5, -1.5],
                [0, 0.5, 0],
                [0, 0, 0],
            ], dtype=out_dtype)
        elif bin_size == (4, 2):
            true_out_arr = np.array([[-0.125, 1.125, 0.375]], dtype=out_dtype)
        else:
            true_out_arr = None

    else:
        assert False, "unknown reduction"

    return bin_size, bin_reduction, bin_dtype, arr, true_out_arr


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


def test_resize_array_fwd(resize_setup, odl_scalar_dtype):
    dtype = odl_scalar_dtype
    pad_mode, pad_const, newshp, offset, array_in, true_out = resize_setup
    array_in = np.array(array_in, dtype=dtype)
    true_out = np.array(true_out, dtype=dtype)

    resized = resize_array(array_in, newshp, offset, pad_mode, pad_const,
                           direction='forward')
    out = np.empty(newshp, dtype=dtype)
    resize_array(array_in, newshp, offset, pad_mode, pad_const,
                 direction='forward', out=out)

    assert np.array_equal(resized, true_out)
    assert np.array_equal(out, true_out)


def test_resize_array_adj(resize_setup, odl_floating_dtype):
    dtype = odl_floating_dtype
    pad_mode, pad_const, newshp, offset, array, _ = resize_setup

    if pad_const != 0:
        # Not well-defined
        return

    array = np.array(array, dtype=dtype)
    if is_real_dtype(dtype):
        other_arr = np.random.uniform(-10, 10, size=newshp)
    else:
        other_arr = (np.random.uniform(-10, 10, size=newshp) +
                     1j * np.random.uniform(-10, 10, size=newshp))

    resized = resize_array(array, newshp, offset, pad_mode, pad_const,
                           direction='forward')
    resized_adj = resize_array(other_arr, array.shape, offset, pad_mode,
                               pad_const, direction='adjoint')

    assert (np.vdot(resized.ravel(), other_arr.ravel()) ==
            pytest.approx(np.vdot(array.ravel(), resized_adj.ravel()),
                          rel=dtype_tol(dtype)))


def test_resize_array_corner_cases(odl_scalar_dtype, padding):
    # Test extreme cases of resizing that are still valid for several
    # `pad_mode`s
    dtype = odl_scalar_dtype
    pad_mode, pad_const = padding

    # Test array
    arr = np.arange(12, dtype=dtype).reshape((3, 4))

    # Resize to and from 0 total size
    squashed_arr = resize_array(arr, (3, 0), pad_mode=pad_mode)
    assert squashed_arr.size == 0

    squashed_arr = resize_array(arr, (0, 0), pad_mode=pad_mode)
    assert squashed_arr.size == 0

    if pad_mode == 'constant':
        # Blowing up from size 0 only works with constant padding
        true_blownup_arr = np.empty_like(arr)
        true_blownup_arr.fill(pad_const)

        blownup_arr = resize_array(np.ones((3, 0), dtype=dtype), (3, 4),
                                   pad_mode=pad_mode, pad_const=pad_const)
        assert np.array_equal(blownup_arr, true_blownup_arr)

        blownup_arr = resize_array(np.ones((0, 0), dtype=dtype), (3, 4),
                                   pad_mode=pad_mode, pad_const=pad_const)
        assert np.array_equal(blownup_arr, true_blownup_arr)

    # Resize from 0 axes to 0 axes
    zero_axes_arr = resize_array(np.array(0, dtype=dtype), (),
                                 pad_mode=pad_mode)
    assert zero_axes_arr == np.array(0, dtype=dtype)

    if pad_mode == 'periodic':
        # Resize with periodic padding, using all values from the original
        # array on both sides
        max_per_shape = (9, 12)
        res_arr = resize_array(arr, max_per_shape, pad_mode='periodic',
                               offset=arr.shape)
        assert np.array_equal(res_arr, np.tile(arr, (3, 3)))

    elif pad_mode == 'symmetric':
        # Symmetric padding, maximum number is one less compared to periodic
        # padding since the boundary value is not repeated
        arr = np.arange(6).reshape((2, 3))
        max_sym_shape = (4, 7)
        res_arr = resize_array(arr, max_sym_shape, pad_mode='symmetric',
                               offset=[1, 2])
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

    # invalid pad mode
    with pytest.raises(ValueError):
        resize_array(arr_1d, (10,), pad_mode='madeup_mode')

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
        resize_array(small_arr, (3, 2), pad_mode='symmetric')
    with pytest.raises(ValueError):
        resize_array(small_arr, (6, 1), pad_mode='symmetric')
    with pytest.raises(ValueError):
        resize_array(small_arr, (4, 3), offset=(0, 1), pad_mode='symmetric')

    # Too large padding sizes for periodic
    small_arr = np.ones((3, 1))
    with pytest.raises(ValueError):
        resize_array(small_arr, (3, 3), pad_mode='periodic')
    with pytest.raises(ValueError):
        resize_array(small_arr, (7, 1), pad_mode='periodic')
    with pytest.raises(ValueError):
        resize_array(small_arr, (3, 4), offset=(0, 1), pad_mode='periodic')


def test_binning(binning_setup):
    """Validate the ``binning`` function."""
    bin_size, bin_reduction, _, arr, true_out_arr = binning_setup

    # Error scenarios
    bin_sizes = bin_size if isinstance(bin_size, tuple) else [bin_size] * 2
    if 0 in bin_sizes:
        # 0 not allowed
        with pytest.raises(ValueError):
            binning(arr, bin_size, bin_reduction)
        return
    if len(bin_sizes) != arr.ndim:
        with pytest.raises(ValueError):
            binning(arr, bin_size, bin_reduction)
    if any(b > n for b, n in zip(bin_sizes, arr.shape)):
        # Bin size larger than array size not allowed
        with pytest.raises(ValueError):
            binning(arr, bin_size, bin_reduction)
        return
    if any(n % b != 0 for b, n in zip(bin_sizes, arr.shape)):
        # Division remainder of size/bin_size not allowed
        with pytest.raises(ValueError):
            binning(arr, bin_size, bin_reduction)
        return

    # Success scenarios
    out_arr = binning(arr, bin_size, bin_reduction)
    assert all_almost_equal(out_arr, true_out_arr)
    assert out_arr.dtype == true_out_arr.dtype


def test_binning_corner_cases():
    # 0-dim
    arr = np.array(1.0)
    assert binning(arr, 2) == 1.0  # ok since there are no axes
    with pytest.raises(ValueError):
        binning(arr, [2])  # not ok, too many binning values

    # 0 in shape, never ok
    arr = np.ones((4, 0))
    with pytest.raises(ValueError):
        binning(arr, 2)
    with pytest.raises(ValueError):
        binning(arr, [1, 1])


if __name__ == '__main__':
    odl.util.test_file(__file__)
