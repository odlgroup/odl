# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for Numpy-based tensors."""

from __future__ import division

import operator

import numpy as np
import pytest

import odl
from odl.set.space import LinearSpaceTypeError
from odl.space.npy_tensors import NumpyTensorSpace
from odl.util.testutils import (
    all_almost_equal, all_equal, noise_array, noise_elements, simple_fixture)

# --- Test helpers --- #


def _pos_array(space):
    """Create an array with positive real entries in ``space``."""
    return np.abs(noise_array(space)) + 0.1


def _array_cls(impl):
    """Return the array class for given impl."""
    if impl == 'numpy':
        return np.ndarray
    else:
        assert False

def _inner(x1, x2, w):
    return np.vdot(x2, w * x1)

def _norm(x, p, w):
    if p in {float('inf'), -float('inf')}:
        return np.linalg.norm(x, p)
    else:
        return np.linalg.norm(w ** (1 / p) * x, p)

def _dist(x1, x2, p, w):
    return _norm(x1 - x2, p, w)


# --- Pytest fixtures --- #


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])

setitem_indices_params = [
    0, [1], (1,), (0, 1), (0, 1, 2), slice(None), slice(None, None, 2),
    (0, slice(None)), (slice(None), 0, slice(None, None, 2))]
setitem_indices = simple_fixture('indices', setitem_indices_params)

getitem_indices_params = (setitem_indices_params +
                          [([0, 1, 1, 0], [0, 1, 1, 2]), (Ellipsis, None)])
getitem_indices = simple_fixture('indices', getitem_indices_params)

weight_params = [1.0, 0.5, _pos_array(odl.tensor_space((3, 4)))]
weight_ids = [' weight=1.0 ', ' weight=0.5 ', ' weight=<array> ']


# scope='module' removed due to pytest issue, see
# https://github.com/pytest-dev/pytest/issues/6497
# TODO(kohr-h): re-introduce (fixed in pytest 5.4.0)
@pytest.fixture(params=weight_params, ids=weight_ids)
def weight(request):
    return request.param


@pytest.fixture(scope='module')
def tspace(odl_floating_dtype, odl_tspace_impl, weight):
    impl = odl_tspace_impl
    dtype = odl_floating_dtype
    return odl.tensor_space(
        shape=(3, 4), dtype=dtype, impl=impl, weighting=weight
    )


# --- Space classes --- #


def test_init_npy_tspace():
    """Test initialization patterns and options for ``NumpyTensorSpace``."""
    # Basic class constructor
    NumpyTensorSpace((3, 4))
    NumpyTensorSpace((3, 4), dtype=int)
    NumpyTensorSpace((3, 4), dtype=float)
    NumpyTensorSpace((3, 4), dtype=complex)
    NumpyTensorSpace((3, 4), dtype=complex, exponent=1.0)
    NumpyTensorSpace((3, 4), dtype=complex, exponent=float('inf'))
    NumpyTensorSpace((3, 4), dtype='S1')

    # Alternative constructor
    odl.tensor_space((3, 4))
    odl.tensor_space((3, 4), dtype=int)
    odl.tensor_space((3, 4), exponent=1.0)

    # Constructors for real spaces
    odl.rn((3, 4))
    odl.rn((3, 4), dtype='float32')
    odl.rn(3)
    odl.rn(3, dtype='float32')

    # Works only for real data types
    with pytest.raises(ValueError):
        odl.rn((3, 4), complex)
    with pytest.raises(ValueError):
        odl.rn(3, int)
    with pytest.raises(ValueError):
        odl.rn(3, 'S1')

    # Constructors for complex spaces
    odl.cn((3, 4))
    odl.cn((3, 4), dtype='complex64')
    odl.cn(3)
    odl.cn(3, dtype='complex64')

    # Works only for complex data types
    with pytest.raises(ValueError):
        odl.cn((3, 4), float)
    with pytest.raises(ValueError):
        odl.cn(3, 'S1')

    # Init with weights or custom space functions
    weight_const = 1.5
    weight_arr = _pos_array(odl.rn((3, 4), float))

    odl.rn((3, 4), weighting=weight_const)
    odl.rn((3, 4), weighting=weight_arr)


def test_init_tspace_weighting(weight, exponent, odl_tspace_impl):
    """Test if weightings during init give the correct weighting classes."""
    impl = odl_tspace_impl
    space = odl.tensor_space((3, 4), weighting=weight, exponent=exponent,
                             impl=impl)

    assert np.all(space.weighting == weight)

    # Errors for bad input
    with pytest.raises(ValueError):
        badly_sized = np.ones((2, 4))
        odl.tensor_space((3, 4), weighting=badly_sized, impl=impl)

    if impl == 'numpy':
        with pytest.raises(ValueError):
            bad_dtype = np.ones((3, 4), dtype=complex)
            odl.tensor_space((3, 4), weighting=bad_dtype)

        with pytest.raises(TypeError):
            odl.tensor_space((3, 4), weighting=1j)  # float() conversion


def test_properties(odl_tspace_impl):
    """Test that the space and element properties are as expected."""
    impl = odl_tspace_impl
    space = odl.tensor_space((3, 4), dtype='float32', exponent=1, weighting=2,
                             impl=impl)
    x = space.element()
    assert x.space is space
    assert x.ndim == space.ndim == 2
    assert x.dtype == space.dtype == np.dtype('float32')
    assert x.size == space.size == 12
    assert x.shape == space.shape == (3, 4)
    assert x.itemsize == 4
    assert x.nbytes == 4 * 3 * 4


def test_size(odl_tspace_impl):
    """Test that size handles corner cases appropriately."""
    impl = odl_tspace_impl
    space = odl.tensor_space((3, 4), impl=impl)
    assert space.size == 12
    assert type(space.size) == int

    # Size 0
    space = odl.tensor_space((), impl=impl)
    assert space.size == 0
    assert type(space.size) == int

    # Overflow test
    large_space = odl.tensor_space((10000,) * 3, impl=impl)
    assert large_space.size == 10000 ** 3
    assert type(space.size) == int


def test_element(tspace, odl_elem_order):
    """Test creation of space elements."""
    order = odl_elem_order
    # From scratch
    elem = tspace.element(order=order)
    assert elem.shape == elem.shape
    assert elem.dtype == tspace.dtype == elem.dtype
    if order is not None:
        assert elem.flags[order + '_CONTIGUOUS']

    # From space elements
    other_elem = tspace.element(np.ones(tspace.shape))
    elem = tspace.element(other_elem, order=order)
    assert all_equal(elem, other_elem)
    if order is None:
        assert elem is other_elem
    else:
        assert elem.flags[order + '_CONTIGUOUS']

    # From Numpy array (C order)
    arr_c = np.random.rand(*tspace.shape).astype(tspace.dtype)
    elem = tspace.element(arr_c, order=order)
    assert all_equal(elem, arr_c)
    assert elem.shape == elem.shape
    assert elem.dtype == tspace.dtype == elem.dtype
    if order is None or order == 'C':
        # None or same order should not lead to copy
        assert np.may_share_memory(elem, arr_c)
    if order is not None:
        # Contiguousness in explicitly provided order should be guaranteed
        assert elem.flags[order + '_CONTIGUOUS']

    # From Numpy array (F order)
    arr_f = np.asfortranarray(arr_c)
    elem = tspace.element(arr_f, order=order)
    assert all_equal(elem, arr_f)
    assert elem.shape == elem.shape
    assert elem.dtype == tspace.dtype == elem.dtype
    if order is None or order == 'F':
        # None or same order should not lead to copy
        assert np.may_share_memory(elem, arr_f)
    if order is not None:
        # Contiguousness in explicitly provided order should be guaranteed
        assert elem.flags[order + '_CONTIGUOUS']

    # From pointer
    arr_c_ptr = arr_c.ctypes.data
    elem = tspace.element(data_ptr=arr_c_ptr, order='C')
    assert all_equal(elem, arr_c)
    assert np.may_share_memory(elem, arr_c)
    arr_f_ptr = arr_f.ctypes.data
    elem = tspace.element(data_ptr=arr_f_ptr, order='F')
    assert all_equal(elem, arr_f)
    assert np.may_share_memory(elem, arr_f)

    # Check errors
    with pytest.raises(ValueError):
        tspace.element(order='A')  # only 'C' or 'F' valid

    with pytest.raises(ValueError):
        tspace.element(data_ptr=arr_c_ptr)  # need order argument

    with pytest.raises(TypeError):
        tspace.element(arr_c, arr_c_ptr)  # forbidden to give both


def test_equals_space(odl_tspace_impl):
    """Test equality check of spaces."""
    impl = odl_tspace_impl
    space = odl.tensor_space(3, impl=impl)
    same_space = odl.tensor_space(3, impl=impl)
    other_space = odl.tensor_space(4, impl=impl)

    assert space == space
    assert space == same_space
    assert space != other_space
    assert hash(space) == hash(same_space)
    assert hash(space) != hash(other_space)


def test_tspace_astype(odl_tspace_impl):
    """Test creation of a space counterpart with new dtype."""
    impl = odl_tspace_impl
    real_space = odl.rn((3, 4), impl=impl)
    int_space = odl.tensor_space((3, 4), dtype=int, impl=impl)
    assert real_space.astype(int) == int_space

    # Test propagation of weightings and the `[real/complex]_space` properties
    real = odl.rn((3, 4), weighting=1.5, impl=impl)
    cplx = odl.cn((3, 4), weighting=1.5, impl=impl)
    real_s = odl.rn((3, 4), weighting=1.5, dtype='float32', impl=impl)
    cplx_s = odl.cn((3, 4), weighting=1.5, dtype='complex64', impl=impl)

    # Real
    assert real.astype('float32') == real_s
    assert real.astype('float64') is real
    assert real.real_space is real
    assert real.astype('complex64') == cplx_s
    assert real.astype('complex128') == cplx
    assert real.complex_space == cplx

    # Complex
    assert cplx.astype('complex64') == cplx_s
    assert cplx.astype('complex128') is cplx
    assert cplx.real_space == real
    assert cplx.astype('float32') == real_s
    assert cplx.astype('float64') == real
    assert cplx.complex_space is cplx


def _test_lincomb(space, a, b, discontig):
    """Validate lincomb against direct result using arrays."""
    # Set slice for discontiguous arrays and get result space of slicing
    if discontig:
        slc = tuple(
            [slice(None)] * (space.ndim - 1) + [slice(None, None, 2)]
        )
        res_space = space.element()[slc].space
    else:
        res_space = space

    # Unaliased arguments
    [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
    if discontig:
        x, y, z = x[slc], y[slc], z[slc]
        xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

    zarr[:] = a * xarr + b * yarr
    res_space.lincomb(a, x, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # First argument aliased with output
    [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
    if discontig:
        x, y, z = x[slc], y[slc], z[slc]
        xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

    zarr[:] = a * zarr + b * yarr
    res_space.lincomb(a, z, b, y, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Second argument aliased with output
    [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
    if discontig:
        x, y, z = x[slc], y[slc], z[slc]
        xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

    zarr[:] = a * xarr + b * zarr
    res_space.lincomb(a, x, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # Both arguments aliased with each other
    [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
    if discontig:
        x, y, z = x[slc], y[slc], z[slc]
        xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

    zarr[:] = a * xarr + b * xarr
    res_space.lincomb(a, x, b, x, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

    # All aliased
    [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
    if discontig:
        x, y, z = x[slc], y[slc], z[slc]
        xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

    zarr[:] = a * zarr + b * zarr
    res_space.lincomb(a, z, b, z, out=z)
    assert all_almost_equal([x, y, z], [xarr, yarr, zarr])


def test_lincomb(tspace):
    """Validate lincomb against direct result using arrays and some scalars."""
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b, discontig=False)


def test_lincomb_discontig(odl_tspace_impl):
    """Test lincomb with discontiguous input."""
    impl = odl_tspace_impl

    scalar_values = [0, 1, -1, 3.41]

    # Use small size for small array case
    tspace = odl.rn((3, 4), impl=impl)

    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b, discontig=True)

    # Use medium size to test fallback impls
    tspace = odl.rn((30, 40), impl=impl)

    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b, discontig=True)


def test_lincomb_raise(tspace):
    """Test if lincomb raises correctly for bad input."""
    other_space = odl.rn((4, 3), impl=tspace.impl)

    other_x = other_space.zero()
    x, y, z = tspace.zero(), tspace.zero(), tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, other_x, 1, y, z)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, y, 1, other_x, z)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, y, 1, z, other_x)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb([], x, 1, y, z)

    with pytest.raises(LinearSpaceTypeError):
        tspace.lincomb(1, x, [], y, z)


def test_multiply(tspace):
    """Test multiply against direct array multiplication."""
    # space method
    [x_arr, y_arr, out_arr], [x, y, out] = noise_elements(tspace, 3)
    out_arr = x_arr * y_arr

    tspace.multiply(x, y, out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])

    # member method
    [x_arr, y_arr, out_arr], [x, y, out] = noise_elements(tspace, 3)
    out_arr = x_arr * y_arr

    x.multiply(y, out=out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])


def test_multiply_exceptions(tspace):
    """Test if multiply raises correctly for bad input."""
    other_space = odl.rn((4, 3))

    other_x = other_space.zero()
    x, y = tspace.zero(), tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.multiply(other_x, x, y)

    with pytest.raises(LinearSpaceTypeError):
        tspace.multiply(x, other_x, y)

    with pytest.raises(LinearSpaceTypeError):
        tspace.multiply(x, y, other_x)


def test_power(tspace):
    """Test ``**`` against direct array exponentiation."""
    [x_arr, y_arr], [x, y] = noise_elements(tspace, n=2)
    y_pos = tspace.element(np.abs(y) + 0.1)
    y_pos_arr = np.abs(y_arr) + 0.1

    # Testing standard positive integer power out-of-place and in-place
    assert all_almost_equal(x ** 2, x_arr ** 2)
    y **= 2
    y_arr **= 2
    assert all_almost_equal(y, y_arr)

    # Real number and negative integer power
    assert all_almost_equal(y_pos ** 1.3, y_pos_arr ** 1.3)
    assert all_almost_equal(y_pos ** (-3), y_pos_arr ** (-3))
    y_pos **= 2.5
    y_pos_arr **= 2.5
    assert all_almost_equal(y_pos, y_pos_arr)

    # Array raised to the power of another array, entry-wise
    assert all_almost_equal(y_pos ** x, y_pos_arr ** x_arr)
    y_pos **= x.real
    y_pos_arr **= x_arr.real
    assert all_almost_equal(y_pos, y_pos_arr)


def test_unary_ops(tspace):
    """Verify that the unary operators (`+x` and `-x`) work as expected."""
    for op in [operator.pos, operator.neg]:
        x_arr, x = noise_elements(tspace)

        y_arr = op(x_arr)
        y = op(x)

        assert all_almost_equal([x, y], [x_arr, y_arr])


def test_scalar_operator(tspace, odl_arithmetic_op):
    """Verify binary operations with scalars.

    Verifies that the statement y = op(x, scalar) gives equivalent results
    to NumPy.
    """
    op = odl_arithmetic_op
    if op in (operator.truediv, operator.itruediv):
        ndigits = int(-np.log10(np.finfo(tspace.dtype).resolution) // 2)
    else:
        ndigits = int(-np.log10(np.finfo(tspace.dtype).resolution))

    for scalar in [-31.2, -1, 0, 1, 2.13]:
        x_arr, x = noise_elements(tspace)

        # Left op
        if scalar == 0 and op in [operator.truediv, operator.itruediv]:
            # Check for correct zero division behaviour
            with pytest.raises(ZeroDivisionError):
                y = op(x, scalar)
        else:
            y_arr = op(x_arr, scalar)
            y = op(x, scalar)

            assert all_almost_equal([x, y], [x_arr, y_arr], ndigits)

        # right op
        x_arr, x = noise_elements(tspace)

        y_arr = op(scalar, x_arr)
        y = op(scalar, x)

        assert all_almost_equal([x, y], [x_arr, y_arr], ndigits)


def test_binary_operator(tspace, odl_arithmetic_op):
    """Verify binary operations with tensors.

    Verifies that the statement z = op(x, y) gives equivalent results
    to NumPy.
    """
    op = odl_arithmetic_op
    if op in (operator.truediv, operator.itruediv):
        ndigits = int(-np.log10(np.finfo(tspace.dtype).resolution) // 2)
    else:
        ndigits = int(-np.log10(np.finfo(tspace.dtype).resolution))

    [x_arr, y_arr], [x, y] = noise_elements(tspace, 2)

    # non-aliased left
    z_arr = op(x_arr, y_arr)
    z = op(x, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], ndigits)

    # non-aliased right
    z_arr = op(y_arr, x_arr)
    z = op(y, x)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], ndigits)

    # aliased operation
    z_arr = op(x_arr, x_arr)
    z = op(x, x)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr], ndigits)


def test_inner(tspace):
    """Test the inner method against numpy.vdot."""
    (xarr, yarr), (x, y) = noise_elements(tspace, 2)
    correct_inner = _inner(xarr, yarr, tspace.weighting)
    assert tspace.inner(x, y) == pytest.approx(correct_inner)


def test_inner_exceptions(tspace):
    """Test if inner raises correctly for bad input."""
    other_space = odl.rn((4, 3))
    other_x = other_space.zero()
    x = tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.inner(other_x, x)

    with pytest.raises(LinearSpaceTypeError):
        tspace.inner(x, other_x)


def test_norm(tspace):
    """Test the norm method against numpy.linalg.norm."""
    xarr, x = noise_elements(tspace)
    correct_norm = _norm(xarr, tspace.exponent, tspace.weighting)
    assert tspace.norm(x) == pytest.approx(correct_norm)


def test_norm_exceptions(tspace):
    """Test if norm raises correctly for bad input."""
    other_space = odl.rn((4, 3))
    other_x = other_space.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.norm(other_x)


def test_pnorm(exponent):
    """Test the norm method with p!=2 against numpy.linalg.norm."""
    for tspace in (odl.rn((3, 4), exponent=exponent),
                   odl.cn((3, 4), exponent=exponent)):
        xarr, x = noise_elements(tspace)
        correct_norm = _norm(xarr, exponent, 1.0)
        assert tspace.norm(x) == pytest.approx(correct_norm)


def test_dist(tspace):
    """Test the dist method against numpy.linalg.norm of the difference."""
    (xarr, yarr), (x, y) = noise_elements(tspace, n=2)
    correct_dist = _dist(xarr, yarr, tspace.exponent, tspace.weighting)
    assert tspace.dist(x, y) == pytest.approx(correct_dist)


def test_dist_exceptions(tspace):
    """Test if dist raises correctly for bad input."""
    other_space = odl.rn((4, 3))
    other_x = other_space.zero()
    x = tspace.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.dist(other_x, x)

    with pytest.raises(LinearSpaceTypeError):
        tspace.dist(x, other_x)


def test_pdist(odl_tspace_impl, exponent):
    """Test the dist method with p!=2 against numpy.linalg.norm of diff."""
    impl = odl_tspace_impl
    spaces = [odl.rn((3, 4), exponent=exponent, impl=impl)]
    cls = odl.space.entry_points.tensor_space_impl(impl)
    if complex in cls.available_dtypes():
        spaces.append(odl.cn((3, 4), exponent=exponent, impl=impl))

    for space in spaces:
        (xarr, yarr), (x, y) = noise_elements(space, n=2)
        correct_dist = _dist(xarr, yarr, exponent, 1.0)
        assert space.dist(x, y) == pytest.approx(correct_dist)


if __name__ == '__main__':
    odl.util.test_file(__file__)
