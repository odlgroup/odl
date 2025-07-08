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
import math
import numpy as np
import pytest

import odl
from odl.set.space import LinearSpaceTypeError
from odl.space.npy_tensors import (
    NumpyTensor, NumpyTensorSpace)
from odl.util.testutils import (
    all_almost_equal, all_equal, noise_array, noise_element, noise_elements,
    isclose, simple_fixture)
from odl.array_API_support import lookup_array_backend
from odl.util.pytest_config import IMPL_DEVICE_PAIRS

from odl.util.dtype_utils import is_complex_dtype

# --- Test helpers --- #

# Functions to return arrays and classes corresponding to impls. Extend
# when a new impl is available.


def _pos_array(space):
    """Create an array with positive real entries in ``space``."""
    ns = space.array_backend.array_namespace
    return ns.abs(noise_array(space)) + 0.1

# --- Pytest fixtures --- #

exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])

setitem_indices_params = [
    0, [1], (1,), (0, 1), (0, 1, 2), slice(None), slice(None, None, 2),
    (0, slice(None)), (slice(None), 0, slice(None, None, 2))]
setitem_indices = simple_fixture('indices', setitem_indices_params)

getitem_indices_params = (setitem_indices_params +
                          [([0, 1, 1, 0], [0, 1, 1, 2]), (Ellipsis, None)])
getitem_indices = simple_fixture('indices', getitem_indices_params)

DEFAULT_SHAPE = (3,4)

@pytest.fixture(scope='module', params=IMPL_DEVICE_PAIRS)
def tspace(request, odl_floating_dtype):
    impl, device = request.param
    dtype = odl_floating_dtype
    return odl.tensor_space(shape=DEFAULT_SHAPE, dtype=dtype, impl=impl, device=device)

@pytest.fixture(scope='module', params=IMPL_DEVICE_PAIRS)
def real_tspace(request, odl_real_floating_dtype):
    impl, device = request.param
    dtype = odl_real_floating_dtype
    return odl.tensor_space(shape=DEFAULT_SHAPE, dtype=dtype, impl=impl, device=device)

# --- Tests --- #
def test_device(odl_impl_device_pairs):
    print(odl_impl_device_pairs)

# def test_init_tspace(odl_tspace_impl, odl_scalar_dtype):
#     constant_weighting = odl.space_weighting(odl_tspace_impl, weight = 1.5)
#     array_weighting    = odl.space_weighting(odl_tspace_impl, weight = _pos_array(odl.rn(DEFAULT_SHAPE)))
#     for device in AVAILABLE_DEVICES[odl_tspace_impl]:
#         for weighting in [constant_weighting, array_weighting, None]:
#             NumpyTensorSpace(DEFAULT_SHAPE, dtype=odl_scalar_dtype, device=device, weighting=weighting)
#             odl.tensor_space(DEFAULT_SHAPE, dtype=odl_scalar_dtype, device=device, weighting=weighting)

# def test_init_tspace_from_cn(odl_tspace_impl, odl_complex_floating_dtype, odl_real_floating_dtype):
#     constant_weighting = odl.space_weighting(odl_tspace_impl, weight = 1.5)
#     array_weighting    = odl.space_weighting(odl_tspace_impl, weight = _pos_array(odl.rn(DEFAULT_SHAPE)))
#     for device in AVAILABLE_DEVICES[odl_tspace_impl]:
#         for weighting in [constant_weighting, array_weighting, None]:
#             odl.cn(DEFAULT_SHAPE, dtype=odl_complex_floating_dtype, device=device, weighting = weighting)
#             with pytest.raises(AssertionError):
#                 odl.cn(DEFAULT_SHAPE, dtype=odl_real_floating_dtype, device=device)
        
# def test_init_tspace_from_rn(odl_tspace_impl, odl_real_floating_dtype, odl_complex_floating_dtype):
#     constant_weighting = odl.space_weighting(odl_tspace_impl, weight = 1.5)
#     array_weighting    = odl.space_weighting(odl_tspace_impl, weight = _pos_array(odl.rn(DEFAULT_SHAPE)))
#     for device in AVAILABLE_DEVICES[odl_tspace_impl]:
#         for weighting in [constant_weighting, array_weighting, None]:
#             odl.rn(DEFAULT_SHAPE, dtype=odl_real_floating_dtype, device=device, weighting = weighting)
#             with pytest.raises(AssertionError):
#                 odl.rn(DEFAULT_SHAPE, dtype=odl_complex_floating_dtype, device=device)

# def test_init_npy_tspace():
#     """Test initialization patterns and options for ``NumpyTensorSpace``."""
#     # Basic class constructor
#     NumpyTensorSpace(DEFAULT_SHAPE)
#     NumpyTensorSpace(DEFAULT_SHAPE, dtype=int)
#     NumpyTensorSpace(DEFAULT_SHAPE, dtype=float)
#     NumpyTensorSpace(DEFAULT_SHAPE, dtype=complex)
#     NumpyTensorSpace(DEFAULT_SHAPE, dtype=complex, exponent=1.0)
#     NumpyTensorSpace(DEFAULT_SHAPE, dtype=complex, exponent=float('inf'))
#     NumpyTensorSpace(DEFAULT_SHAPE, dtype='S1')

#     # Alternative constructor
#     odl.tensor_space(DEFAULT_SHAPE)
#     odl.tensor_space(DEFAULT_SHAPE, dtype=int)
#     odl.tensor_space(DEFAULT_SHAPE, exponent=1.0)

#     # Constructors for real spaces
#     odl.rn(DEFAULT_SHAPE)
#     odl.rn(DEFAULT_SHAPE, dtype='float32')
#     odl.rn(3)
#     odl.rn(3, dtype='float32')

#     # Works only for real data types
#     with pytest.raises(ValueError):
#         odl.rn(DEFAULT_SHAPE, complex)
#     with pytest.raises(ValueError):
#         odl.rn(3, int)
#     with pytest.raises(ValueError):
#         odl.rn(3, 'S1')

#     # Constructors for complex spaces
#     odl.cn(DEFAULT_SHAPE)
#     odl.cn(DEFAULT_SHAPE, dtype='complex64')
#     odl.cn(3)
#     odl.cn(3, dtype='complex64')

#     # Works only for complex data types
#     with pytest.raises(ValueError):
#         odl.cn(DEFAULT_SHAPE, float)
#     with pytest.raises(ValueError):
#         odl.cn(3, 'S1')

#     # Init with weights or custom space functions
#     weight_const = 1.5
#     weight_arr = _pos_array(odl.rn(DEFAULT_SHAPE, float))

#     odl.rn(DEFAULT_SHAPE, weighting=weight_const)
#     odl.rn(DEFAULT_SHAPE, weighting=weight_arr)


# def test_init_tspace_weighting(exponent, odl_tspace_impl, odl_scalar_dtype):
#     """Test if weightings during init give the correct weighting classes."""
#     impl = odl_tspace_impl

#     for device in AVAILABLE_DEVICES[impl]:
#         weight_params = [1, 0.5, _pos_array(odl.rn(DEFAULT_SHAPE, impl=impl, device=device))]
#         for weight in weight_params:
#             # We compare that a space instanciated with a given weight has its weight
#             # equal to the weight of a weighting class instanciated through odl.space_weighting
#             weighting = odl.space_weighting(                
#                 weight=weight, exponent=exponent, impl=impl, device=device)
            
#             space = odl.tensor_space(
#                 DEFAULT_SHAPE, dtype=odl_scalar_dtype,weight=weight, exponent=exponent, impl=impl, device=device)

#             assert space.weighting == weighting

#         with pytest.raises(ValueError):
#             badly_sized = odl.space_weighting(
#                 impl=impl, device=device,
#                 weight = np.ones((2, 4)), exponent=exponent)
#             odl.tensor_space(DEFAULT_SHAPE, weighting=badly_sized, impl=impl)


def test_properties(odl_tspace_impl):
    """Test that the space and element properties are as expected."""
    impl = odl_tspace_impl
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2,
                             impl=impl)
    x = space.element()
    assert x.space is space
    assert x.ndim == space.ndim == 2
    assert x.dtype == space.dtype == np.dtype('float32')
    assert x.size == space.size == 12
    assert x.shape == space.shape == DEFAULT_SHAPE
    assert x.itemsize == 4
    assert x.nbytes == 4 * 3 * 4
    assert x.device == 'cpu'


def test_size(odl_tspace_impl, odl_scalar_dtype):
    """Test that size handles corner cases appropriately."""
    impl = odl_tspace_impl
    space = odl.tensor_space(DEFAULT_SHAPE,dtype=odl_scalar_dtype,  impl=impl)
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


# Test deprecated as we assume the order to be C contiguous and 
# we can't create an element from a pointer anymore
# def test_element(tspace, odl_elem_order):
#     """Test creation of space elements."""
#     order = odl_elem_order
#     # From scratch
#     elem = tspace.element(order=order)
#     assert elem.shape == elem.data.shape
#     assert elem.dtype == tspace.dtype == elem.data.dtype
#     if order is not None:
#         assert elem.data.flags[order + '_CONTIGUOUS']

#     # From space elements
#     other_elem = tspace.element(np.ones(tspace.shape))
#     elem = tspace.element(other_elem, order=order)
#     assert all_equal(elem, other_elem)
#     if order is None:
#         assert elem is other_elem
#     else:
#         assert elem.data.flags[order + '_CONTIGUOUS']

#     # From Numpy array (C order)
#     arr_c = np.random.rand(*tspace.shape).astype(tspace.dtype)
#     elem = tspace.element(arr_c, order=order)
#     assert all_equal(elem, arr_c)
#     assert elem.shape == elem.data.shape
#     assert elem.dtype == tspace.dtype == elem.data.dtype
#     if order is None or order == 'C':
#         # None or same order should not lead to copy
#         assert np.may_share_memory(elem.data, arr_c)
#     if order is not None:
#         # Contiguousness in explicitly provided order should be guaranteed
#         assert elem.data.flags[order + '_CONTIGUOUS']

#     # From Numpy array (F order)
#     arr_f = np.asfortranarray(arr_c)
#     elem = tspace.element(arr_f, order=order)
#     assert all_equal(elem, arr_f)
#     assert elem.shape == elem.data.shape
#     assert elem.dtype == tspace.dtype == elem.data.dtype
#     if order is None or order == 'F':
#         # None or same order should not lead to copy
#         assert np.may_share_memory(elem.data, arr_f)
#     if order is not None:
#         # Contiguousness in explicitly provided order should be guaranteed
#         assert elem.data.flags[order + '_CONTIGUOUS']

#     # From pointer
#     arr_c_ptr = arr_c.ctypes.data
#     elem = tspace.element(data_ptr=arr_c_ptr, order='C')
#     assert all_equal(elem, arr_c)
#     assert np.may_share_memory(elem.data, arr_c)
#     arr_f_ptr = arr_f.ctypes.data
#     elem = tspace.element(data_ptr=arr_f_ptr, order='F')
#     assert all_equal(elem, arr_f)
#     assert np.may_share_memory(elem.data, arr_f)

#     # Check errors
#     with pytest.raises(ValueError):
#         tspace.element(order='A')  # only 'C' or 'F' valid

#     with pytest.raises(ValueError):
#         tspace.element(data_ptr=arr_c_ptr)  # need order argument

#     with pytest.raises(TypeError):
#         tspace.element(arr_c, arr_c_ptr)  # forbidden to give both


# def test_equals_space(odl_tspace_impl, odl_scalar_dtype):
#     """Test equality check of spaces."""
#     impl = odl_tspace_impl
#     for device in AVAILABLE_DEVICES[impl]:
#         space = odl.tensor_space(3, impl=impl, dtype=odl_scalar_dtype, device=device)
#         same_space = odl.tensor_space(3, impl=impl, dtype=odl_scalar_dtype, device=device)
#         other_space = odl.tensor_space(4, impl=impl, dtype=odl_scalar_dtype, device=device)

#         assert space == space
#         assert space == same_space
#         assert space != other_space
#         assert hash(space) == hash(same_space)
#         assert hash(space) != hash(other_space)


def test_equals_elem(odl_tspace_impl):
    """Test equality check of space elements."""
    impl = odl_tspace_impl
    r3 = odl.rn(3, exponent=2, impl=impl)
    r3_1 = odl.rn(3, exponent=1, impl=impl)
    r4 = odl.rn(4, exponent=2, impl=impl)
    r3_elem = r3.element([1, 2, 3])
    r3_same_elem = r3.element([1, 2, 3])
    r3_other_elem = r3.element([2, 2, 3])
    r3_1_elem = r3_1.element([1, 2, 3])
    r4_elem = r4.element([1, 2, 3, 4])

    assert r3_elem == r3_elem
    assert r3_elem == r3_same_elem
    assert r3_elem != r3_other_elem
    assert r3_elem != r3_1_elem
    assert r3_elem != r4_elem


def test_tspace_astype(odl_tspace_impl):
    """Test creation of a space counterpart with new dtype."""
    impl = odl_tspace_impl
    real_space = odl.rn(DEFAULT_SHAPE, impl=impl)
    int_space = odl.tensor_space(DEFAULT_SHAPE, dtype=int, impl=impl)
    assert real_space.astype(int) == int_space

    # Test propagation of weightings and the `[real/complex]_space` properties
    real = odl.rn(DEFAULT_SHAPE, weighting=1.5, impl=impl)
    cplx = odl.cn(DEFAULT_SHAPE, weighting=1.5, impl=impl)
    real_s = odl.rn(DEFAULT_SHAPE, weighting=1.5, dtype='float32', impl=impl)
    cplx_s = odl.cn(DEFAULT_SHAPE, weighting=1.5, dtype='complex64', impl=impl)

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


# def _test_lincomb(space, a, b, discontig):
#     """Validate lincomb against direct result using arrays."""
#     # Set slice for discontiguous arrays and get result space of slicing
#     # What the actual fuck
#     if discontig:
#         slc = tuple(
#             [slice(None)] * (space.ndim - 1) + [slice(None, None, 2)]
#         )
#         res_space = space.element()[slc].space
#     else:
#         res_space = space

#     # Unaliased arguments
#     [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
#     if discontig:
#         x, y, z = x[slc], y[slc], z[slc]
#         xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

#     zarr[:] = a * xarr + b * yarr
#     res_space.lincomb(a, x, b, y, out=z)
#     assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

#     # First argument aliased with output
#     [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
#     if discontig:
#         x, y, z = x[slc], y[slc], z[slc]
#         xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

#     zarr[:] = a * zarr + b * yarr
#     res_space.lincomb(a, z, b, y, out=z)
#     assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

#     # Second argument aliased with output
#     [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
#     if discontig:
#         x, y, z = x[slc], y[slc], z[slc]
#         xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

#     zarr[:] = a * xarr + b * zarr
#     res_space.lincomb(a, x, b, z, out=z)
#     assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

#     # Both arguments aliased with each other
#     [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
#     if discontig:
#         x, y, z = x[slc], y[slc], z[slc]
#         xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

#     zarr[:] = a * xarr + b * xarr
#     res_space.lincomb(a, x, b, x, out=z)
#     assert all_almost_equal([x, y, z], [xarr, yarr, zarr])

#     # All aliased
#     [xarr, yarr, zarr], [x, y, z] = noise_elements(space, 3)
#     if discontig:
#         x, y, z = x[slc], y[slc], z[slc]
#         xarr, yarr, zarr = xarr[slc], yarr[slc], zarr[slc]

#     zarr[:] = a * zarr + b * zarr
#     res_space.lincomb(a, z, b, z, out=z)
#     assert all_almost_equal([x, y, z], [xarr, yarr, zarr])


# def test_lincomb(tspace):
#     """Validate lincomb against direct result using arrays and some scalars."""
#     scalar_values = [0, 1, -1, 3.41]
#     for a in scalar_values:
#         for b in scalar_values:
#             _test_lincomb(tspace, a, b, discontig=False)


# def test_lincomb_discontig(odl_tspace_impl):
#     """Test lincomb with discontiguous input."""
#     impl = odl_tspace_impl

#     scalar_values = [0, 1, -1, 3.41]

#     # Use small size for small array case
#     tspace = odl.rn(DEFAULT_SHAPE, impl=impl)

#     for a in scalar_values:
#         for b in scalar_values:
#             _test_lincomb(tspace, a, b, discontig=True)

#     # Use medium size to test fallback impls
#     tspace = odl.rn((30, 40), impl=impl)

#     for a in scalar_values:
#         for b in scalar_values:
#             _test_lincomb(tspace, a, b, discontig=True)


# def test_lincomb_exceptions(tspace):
#     """Test whether lincomb raises correctly for bad output element."""
#     other_space = odl.rn((4, 3), impl=tspace.impl)

#     other_x = other_space.zero()
#     x, y, z = tspace.zero(), tspace.zero(), tspace.zero()

#     with pytest.raises(LinearSpaceTypeError):
#         tspace.lincomb(1, other_x, 1, y, z)

#     with pytest.raises(LinearSpaceTypeError):
#         tspace.lincomb(1, y, 1, other_x, z)

#     with pytest.raises(LinearSpaceTypeError):
#         tspace.lincomb(1, y, 1, z, other_x)

#     with pytest.raises(LinearSpaceTypeError):
#         tspace.lincomb([], x, 1, y, z)

#     with pytest.raises(LinearSpaceTypeError):
#         tspace.lincomb(1, x, [], y, z)


def test_multiply__(tspace):
    """Test multiply against direct array multiplication."""
    # space method
    # for device in IMPL_DEVICES[odl_tspace_impl]:
    #     tspace = odl.tensor_space(DEFAULT_SHAPE, dtype=odl_scalar_dtype, impl=odl_tspace_impl, device=device)
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

    with pytest.raises(AssertionError):
        tspace.multiply(other_x, x, y)

    with pytest.raises(AssertionError):
        tspace.multiply(x, other_x, y)

    with pytest.raises(AssertionError):
        tspace.multiply(x, y, other_x)


def test_power(tspace):
    """Test ``**`` against direct array exponentiation."""
    [x_arr, y_arr], [x, y] = noise_elements(tspace, n=2)
    y_pos = tspace.element(odl.abs(y) + 0.1)
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
        ndigits = int(-tspace.array_namespace.log10(tspace.finfo().resolution) // 2)
    else:
        ndigits = int(-tspace.array_namespace.log10(tspace.finfo().resolution))

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
        ndigits = int(-tspace.array_namespace.log10(tspace.finfo().resolution) // 2)
    else:
        ndigits = int(-tspace.array_namespace.log10(tspace.finfo().resolution))

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


def test_assign(tspace):
    """Test the assign method using ``==`` comparison."""
    x = noise_element(tspace)
    x_old = x
    y = noise_element(tspace)

    y.assign(x)

    assert y == x
    assert y is not x
    assert x is x_old

    # test alignment
    x *= 2
    assert y != x


def test_inner(tspace):
    """Test the inner method against numpy.vdot."""
    xarr, xd = noise_elements(tspace)
    yarr, yd = noise_elements(tspace)

    # TODO: add weighting
    correct_inner = np.vdot(yarr, xarr)
    assert tspace.inner(xd, yd) == pytest.approx(correct_inner)
    assert xd.inner(yd) == pytest.approx(correct_inner)


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
    xarr, x = noise_elements(tspace)

    correct_norm = np.linalg.norm(xarr.ravel())

    array_backend = tspace.array_backend
    real_dtype = array_backend.identifier_of_dtype(tspace.real_dtype)
    if real_dtype == "float16":
        tolerance = 5e-3
    elif real_dtype == "float32":
        tolerance = 5e-7
    elif real_dtype == "float64" or real_dtype == float:
        tolerance = 1e-15
    elif real_dtype == "float128":
        tolerance = 1e-19
    else:
        raise TypeError(f"No known tolerance for dtype {tspace.dtype}")
    
    assert tspace.norm(x) == pytest.approx(correct_norm, rel=tolerance)
    assert x.norm() == pytest.approx(correct_norm, rel=tolerance)


    correct_norm = np.linalg.norm(xarr.ravel())

def test_norm_exceptions(tspace):
    """Test if norm raises correctly for bad input."""
    other_space = odl.rn((4, 3))
    other_x = other_space.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.norm(other_x)


def test_pnorm(exponent):
    """Test the norm method with p!=2 against numpy.linalg.norm."""
    for tspace in (odl.rn(DEFAULT_SHAPE, exponent=exponent),
                   odl.cn(DEFAULT_SHAPE, exponent=exponent)):
        xarr, x = noise_elements(tspace)
        correct_norm = np.linalg.norm(xarr.ravel(), ord=exponent)

        assert tspace.norm(x) == pytest.approx(correct_norm)
        assert x.norm() == pytest.approx(correct_norm)


def test_dist(tspace):
    """Test the dist method against numpy.linalg.norm of the difference."""

    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    correct_dist = np.linalg.norm((xarr - yarr).ravel())

    array_backend = tspace.array_backend
    real_dtype = array_backend.identifier_of_dtype(tspace.real_dtype)

    if real_dtype == "float16":
        tolerance = 5e-3
    elif real_dtype == "float32":
        tolerance = 5e-7
    elif real_dtype == "float64" or real_dtype == float:
        tolerance = 1e-15
    elif real_dtype == "float128":
        tolerance = 1e-19

    else:
        raise TypeError(f"No known tolerance for dtype {tspace.dtype}")
    
    assert tspace.dist(x, y) == pytest.approx(correct_dist, rel=tolerance)
    assert x.dist(y) == pytest.approx(correct_dist, rel=tolerance)



# def test_dist_exceptions(odl_tspace_impl):
#     """Test if dist raises correctly for bad input."""
#     for device in AVAILABLE_DEVICES[odl_tspace_impl]:
#         tspace = odl.tensor_space(DEFAULT_SHAPE, impl=odl_tspace_impl, device=device)
#         other_space = odl.rn((4, 3))
#         other_x = other_space.zero()
#         x = tspace.zero()

#         with pytest.raises(LinearSpaceTypeError):
#             tspace.dist(other_x, x)

#         with pytest.raises(LinearSpaceTypeError):
#             tspace.dist(x, other_x)


def test_pdist(odl_tspace_impl, exponent):
    """Test the dist method with p!=2 against numpy.linalg.norm of diff."""
    impl = odl_tspace_impl
    spaces = [
        odl.rn(DEFAULT_SHAPE, exponent=exponent, impl=impl),
        odl.cn(DEFAULT_SHAPE, exponent=exponent, impl=impl)
        ]
    # cls = odl.space.entry_points.tensor_space_impl(impl)

    # if complex in cls.available_dtypes:
    #     spaces.append(odl.cn(DEFAULT_SHAPE, exponent=exponent, impl=impl))

    for space in spaces:
        [xarr, yarr], [x, y] = noise_elements(space, n=2)

        correct_dist = np.linalg.norm((xarr - yarr).ravel(), ord=exponent)
        assert space.dist(x, y) == pytest.approx(correct_dist)
        assert x.dist(y) == pytest.approx(correct_dist)


def test_element_getitem(odl_tspace_impl, getitem_indices):
    """Check if getitem produces correct values, shape and other stuff."""
    impl = odl_tspace_impl
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl)
    x_arr, x = noise_elements(space)

    x_arr_sliced = x_arr[getitem_indices]
    sliced_shape = x_arr_sliced.shape
    x_sliced = x[getitem_indices]

    if np.isscalar(x_arr_sliced):
        assert x_arr_sliced == x_sliced
    else:
        assert x_sliced.shape == sliced_shape
        assert all_equal(x_sliced, x_arr_sliced)

        # Check that the space properties are preserved
        sliced_spc = x_sliced.space
        assert sliced_spc.shape == sliced_shape
        assert sliced_spc.dtype == space.dtype
        assert sliced_spc.exponent == space.exponent
        assert sliced_spc.weighting == space.weighting

        # Check that we have a view that manipulates the original array
        # (or not, depending on indexing style)
        x_arr_sliced[:] = 0
        x_sliced[:] = 0
        assert all_equal(x_arr, x)


def test_element_setitem(odl_tspace_impl, setitem_indices):
    """Check if setitem produces the same result as NumPy."""
    impl = odl_tspace_impl
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl)
    x_arr, x = noise_elements(space)

    x_arr_sliced = x_arr[setitem_indices]
    sliced_shape = x_arr_sliced.shape

    # Setting values with scalars
    x_arr[setitem_indices] = 2.3
    x[setitem_indices] = 2.3
    assert all_equal(x, x_arr)

    # Setting values with arrays
    rhs_arr = np.ones(sliced_shape)
    x_arr[setitem_indices] = rhs_arr
    x[setitem_indices] = rhs_arr
    assert all_equal(x, x_arr)

    # Using a list of lists
    rhs_list = (-np.ones(sliced_shape)).tolist()
    x_arr[setitem_indices] = rhs_list
    x[setitem_indices] = rhs_list
    assert all_equal(x, x_arr)


def test_element_getitem_bool_array(odl_tspace_impl):
    """Check if getitem with boolean array yields the same result as NumPy."""
    impl = odl_tspace_impl
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl)
    bool_space = odl.tensor_space((2, 3, 4), dtype=bool)
    x_arr, x = noise_elements(space)
    cond_arr, cond = noise_elements(bool_space)

    x_arr_sliced = x_arr[cond_arr]
    x_sliced = x[cond]
    assert all_equal(x_arr_sliced, x_sliced)

    # Check that the space properties are preserved
    sliced_spc = x_sliced.space
    assert sliced_spc.shape == x_arr_sliced.shape
    assert sliced_spc.dtype == space.dtype
    assert sliced_spc.exponent == space.exponent
    assert sliced_spc.weighting == space.weighting


def test_element_setitem_bool_array(odl_tspace_impl):
    """Check if setitem produces the same result as NumPy."""
    impl = odl_tspace_impl
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl)
    bool_space = odl.tensor_space((2, 3, 4), dtype=bool)
    x_arr, x = noise_elements(space)
    cond_arr, cond = noise_elements(bool_space)

    x_arr_sliced = x_arr[cond_arr]
    sliced_shape = x_arr_sliced.shape

    # Setting values with scalars
    x_arr[cond_arr] = 2.3
    x[cond] = 2.3
    assert all_equal(x, x_arr)

    # Setting values with arrays
    rhs_arr = np.ones(sliced_shape)
    x_arr[cond_arr] = rhs_arr
    x[cond] = rhs_arr
    assert all_equal(x, x_arr)

    # Using a list of lists
    rhs_list = (-np.ones(sliced_shape)).tolist()
    x_arr[cond_arr] = rhs_list
    x[cond] = rhs_list
    assert all_equal(x, x_arr)


def test_transpose(odl_tspace_impl):
    """Test the .T property of tensors against plain inner product."""
    impl = odl_tspace_impl
    spaces = [
        odl.rn(DEFAULT_SHAPE, impl=impl),
        odl.cn(DEFAULT_SHAPE, impl=impl)
        ]
    # cls = odl.space.entry_points.tensor_space_impl(impl)
    # if complex in cls.available_dtypes():
    #     spaces.append(odl.cn(DEFAULT_SHAPE, impl=impl))

    for space in spaces:
        x = noise_element(space)
        y = noise_element(space)

        # Assert linear operator
        assert isinstance(x.T, odl.Operator)
        assert x.T.is_linear

        # Check result
        assert x.T(y) == pytest.approx(y.inner(x))
        assert all_equal(x.T.adjoint(1.0), x)

        # x.T.T returns self
        assert x.T.T == x

# TODO: SHOULD that be supported???
def test_multiply_by_scalar(tspace):
    """Verify that mult. with NumPy scalars preserves the element type."""
    x = tspace.zero()

    # Simple scalar multiplication, as often performed in user code.
    # This invokes the __mul__ and __rmul__ methods of the ODL space classes.
    # Strictly speaking this operation loses precision if `tspace.dtype` has
    # fewer than 64 bits (Python decimal literals are double precision), but
    # it would be too cumbersome to force a change in the space's dtype.
    assert x * 1.0 in tspace
    assert 1.0 * x in tspace
    
    # Multiplying with NumPy scalars is (since NumPy-2) more restrictive:
    # multiplying a scalar on the left that has a higher precision than can
    # be represented in the space would upcast `x` to another space that has
    # the required precision.
    # This should not be supported anymore
    # if np.can_cast(np.float32, tspace.dtype):
    #     assert x * np.float32(1.0) in tspace
    #     assert np.float32(1.0) * x in tspace


def test_member_copy(odl_tspace_impl):
    """Test copy method of elements."""
    impl = odl_tspace_impl
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2,
                             impl=impl)
    x = noise_element(space)

    y = x.copy()
    assert x == y
    assert y is not x

    # Check that result is not aliased
    x *= 2
    assert x != y


def test_python_copy(odl_tspace_impl):
    """Test compatibility with the Python copy module."""
    import copy
    impl = odl_tspace_impl
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2,
                             impl=impl)
    x = noise_element(space)

    # Shallow copy
    y = copy.copy(x)
    assert x == y
    assert y is not x

    # Check that result is not aliased
    x *= 2
    assert x != y

    # Deep copy
    z = copy.deepcopy(x)
    assert x == z
    assert z is not x

    # Check that result is not aliased
    x *= 2
    assert x != z


def test_conversion_to_scalar(odl_tspace_impl):
    """Test conversion of size-1 vectors/tensors to scalars."""
    impl = odl_tspace_impl
    space = odl.rn(1, impl=impl)
    # Size 1 real space
    value = 1.5
    element = space.element(value)

    assert int(element) == int(value)
    assert float(element) == float(value)
    assert complex(element) == complex(value)

    # Size 1 complex space
    value = 1.5 + 0.5j
    element = odl.cn(1).element(value)
    assert complex(element) == complex(value)

    # Size 1 multi-dimensional space
    value = 2.1
    element = odl.rn((1, 1, 1)).element(value)
    assert float(element) == float(value)

    # Too large space
    element = odl.rn(2).one()

    with pytest.raises(ValueError):
        int(element)
    with pytest.raises(ValueError):
        float(element)
    with pytest.raises(ValueError):
        complex(element)

def test_bool_conversion(odl_tspace_impl):
    """Verify that the __bool__ function works."""
    impl = odl_tspace_impl
    space = odl.tensor_space(2, dtype='float32', impl=impl)
    x = space.element([0, 1])

    with pytest.raises(ValueError):
        bool(x)
    assert odl.any(x)
    assert any(x)
    assert not odl.all(x)
    assert not all(x)

    space = odl.tensor_space(1, dtype='float32', impl=impl)
    x = space.one()

    assert odl.any(x)
    assert any(x)
    assert odl.all(x)
    assert all(x)


# def test_numpy_array_interface(odl_tspace_impl):
#     """Verify that the __array__ interface for NumPy works."""
#     impl = odl_tspace_impl
#     space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2,
#                              impl=impl)
#     x = space.one()
#     arr = x.__array__()

#     assert isinstance(arr, np.ndarray)
#     assert np.array_equal(arr, np.ones(x.shape))

#     x_arr = np.array(x)
#     assert np.array_equal(x_arr, np.ones(x.shape))
#     x_as_arr = np.asarray(x)
#     assert np.array_equal(x_as_arr, np.ones(x.shape))
#     x_as_any_arr = np.asanyarray(x)
#     assert np.array_equal(x_as_any_arr, np.ones(x.shape))


def test_array_wrap_method(odl_tspace_impl):
    """Verify that the __array_wrap__ method for NumPy works."""
    impl = odl_tspace_impl
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2,
                             impl=impl)
    x_arr, x = noise_elements(space)
    y_arr = space.array_namespace.sin(x_arr)
    y = odl.sin(x)  # Should yield again an ODL tensor

    assert all_equal(y, y_arr)
    assert y in space


def test_conj(tspace):
    """Test complex conjugation of tensors."""
    xarr, x = noise_elements(tspace)

    xconj = x.conj()
    assert all_equal(xconj, xarr.conj())

    y = tspace.element()
    xconj = x.conj(out=y)
    assert xconj is y
    assert all_equal(y, xarr.conj())


# --- Weightings (Numpy) --- #


def test_array_weighting_init(real_tspace):
    """Test initialization of array weightings."""
    exponent = 2
    array_backend = real_tspace.array_backend
    impl = real_tspace.impl
    weight_arr = _pos_array(real_tspace)
    weight_elem = real_tspace.element(weight_arr)

    weighting_arr  = odl.space_weighting(impl, device=real_tspace.device, weight=weight_arr, exponent=exponent)
    weighting_elem = odl.space_weighting(impl, device=real_tspace.device, 
    weight=weight_elem, exponent=exponent)

    assert isinstance(weighting_arr.weight, array_backend.array_type)
    assert isinstance(weighting_elem.weight, array_backend.array_type)


def test_array_weighting_array_is_valid(odl_tspace_impl):
    """Test the is_valid method of array weightings."""
    impl = odl_tspace_impl
    space = odl.rn(DEFAULT_SHAPE, impl=impl)
    weight_arr = _pos_array(space)

    assert odl.space_weighting(impl, weight=weight_arr)
    # Invalid
    weight_arr[0] = 0
    with pytest.raises(ValueError):
        odl.space_weighting(impl, weight=weight_arr)


def test_array_weighting_equals(odl_tspace_impl):
    """Test the equality check method of array weightings."""
    impl = odl_tspace_impl
    space = odl.rn(5, impl=impl)
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)

    weighting_arr = odl.space_weighting(odl_tspace_impl, weight=weight_arr)
    weighting_arr2 = odl.space_weighting(odl_tspace_impl, weight=weight_arr)
    weighting_elem = odl.space_weighting(odl_tspace_impl, weight=weight_elem)
    weighting_elem_copy = odl.space_weighting(odl_tspace_impl, weight=weight_elem.copy())
    weighting_elem2 = odl.space_weighting(odl_tspace_impl, weight=weight_elem)
    weighting_other_arr = odl.space_weighting(odl_tspace_impl, weight=weight_arr +1 )
    weighting_other_exp = odl.space_weighting(odl_tspace_impl, weight=weight_arr +1, exponent=1)

    assert weighting_arr == weighting_arr2
    assert weighting_arr == weighting_elem
    assert weighting_arr == weighting_elem_copy
    assert weighting_elem == weighting_elem2
    assert weighting_arr != weighting_other_arr
    assert weighting_arr != weighting_other_exp


def test_array_weighting_equiv(odl_tspace_impl):
    """Test the equiv method of Numpy array weightings."""
    impl = odl_tspace_impl
    space = odl.rn(5, impl=impl)
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)
    different_arr = weight_arr + 1
    w_arr = odl.space_weighting(odl_tspace_impl, weight=weight_arr)
    w_elem = odl.space_weighting(odl_tspace_impl, weight=weight_elem)
    w_different_arr = odl.space_weighting(odl_tspace_impl, weight=different_arr)

    # Equal -> True
    assert w_arr.equiv(w_arr)
    assert w_arr.equiv(w_elem)
    # Different array -> False
    assert not w_arr.equiv(w_different_arr)

    # Test shortcuts in the implementation
    const_arr = np.ones(space.shape) * 1.5
    w_const_arr = odl.space_weighting(odl_tspace_impl, weight=const_arr)
    w_const = odl.space_weighting(odl_tspace_impl, weight=1.5)
    w_wrong_const = odl.space_weighting(odl_tspace_impl, weight=1)
    w_wrong_exp = odl.space_weighting(odl_tspace_impl, weight=1.5, exponent=1)

    assert w_const_arr.equiv(w_const)
    assert not w_const_arr.equiv(w_wrong_const)
    assert not w_const_arr.equiv(w_wrong_exp)

    # Bogus input
    assert not w_const_arr.equiv(True)
    assert not w_const_arr.equiv(object)
    assert not w_const_arr.equiv(None)


def test_array_weighting_inner(tspace):
    """Test inner product in a weighted space."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    weight_arr = _pos_array(tspace)
    weighting = odl.space_weighting(impl = tspace.impl, weight = weight_arr)

    true_inner = np.vdot(yarr, xarr * weight_arr)
    assert weighting.inner(x.data, y.data) == pytest.approx(true_inner)

    # Exponent != 2 -> no inner product, should raise
    with pytest.raises(NotImplementedError):
        odl.space_weighting(impl = tspace.impl, weight =weight_arr, exponent=1.0).inner(x.data, y.data)


def test_array_weighting_norm(tspace, exponent):
    """Test norm in a weighted space."""
    ns = tspace.array_namespace
    rtol = ns.sqrt(ns.finfo(tspace.dtype).resolution)
    xarr, x = noise_elements(tspace)

    weight_arr = _pos_array(tspace)
    weighting = odl.space_weighting(impl = tspace.impl, weight=weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_norm = ns.linalg.vector_norm(
            weight_arr * xarr,
            ord=exponent)
    else:
        true_norm = ns.linalg.norm(
            (weight_arr ** (1 / exponent) * xarr).ravel(),
            ord=exponent)

    assert weighting.norm(x.data) == pytest.approx(true_norm, rel=rtol)


def test_array_weighting_dist(tspace, exponent):
    """Test dist product in a weighted space."""
    ns = tspace.array_namespace
    rtol = ns.sqrt(ns.finfo(tspace.dtype).resolution)
    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    weight_arr = _pos_array(tspace)
    weighting = odl.space_weighting(impl = tspace.impl, weight=weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_dist = np.linalg.norm(
            (weight_arr * (xarr - yarr)).ravel(),
            ord=float('inf'))
    else:
        true_dist = np.linalg.norm(
            (weight_arr ** (1 / exponent) * (xarr - yarr)).ravel(),
            ord=exponent)

    assert weighting.dist(x.data, y.data) == pytest.approx(true_dist, rel=rtol)


def test_const_weighting_init(odl_tspace_impl, exponent):
    """Test initialization of constant weightings."""

    # Just test if the code runs
    odl.space_weighting(impl=odl_tspace_impl, weight=1.5, exponent=exponent)
    with pytest.raises(ValueError):
        odl.space_weighting(impl=odl_tspace_impl, weight=0, exponent=exponent)
    with pytest.raises(ValueError):
        odl.space_weighting(impl=odl_tspace_impl, weight=-1.5, exponent=exponent)
    with pytest.raises(ValueError):
        odl.space_weighting(impl=odl_tspace_impl, weight=float('inf'), exponent=exponent)


def test_const_weighting_comparison(odl_tspace_impl):
    """Test equality to and equivalence with const weightings."""
    impl = odl_tspace_impl
    constant = 1.5

    w_const = odl.space_weighting(impl=odl_tspace_impl, weight=constant)
    w_const2 = odl.space_weighting(impl=odl_tspace_impl, weight=constant)
    w_other_const = odl.space_weighting(impl=odl_tspace_impl, weight=constant+1)
    w_other_exp = odl.space_weighting(impl=odl_tspace_impl, weight=constant, exponent = 1)

    const_arr = constant * np.ones(DEFAULT_SHAPE)

    w_const_arr = odl.space_weighting(impl=odl_tspace_impl, weight=const_arr)
    other_const_arr = (constant + 1) * np.ones(DEFAULT_SHAPE)
    w_other_const_arr =  odl.space_weighting(impl=odl_tspace_impl, weight=other_const_arr)

    assert w_const == w_const
    assert w_const == w_const2
    assert w_const2 == w_const
    # Different but equivalent
    assert w_const.equiv(w_const_arr)
    assert w_const != w_const_arr

    # Not equivalent
    assert not w_const.equiv(w_other_exp)
    assert w_const != w_other_exp
    assert not w_const.equiv(w_other_const)
    assert w_const != w_other_const
    assert not w_const.equiv(w_other_const_arr)
    assert w_const != w_other_const_arr

    # Bogus input
    assert not w_const.equiv(True)
    assert not w_const.equiv(object)
    assert not w_const.equiv(None)


def test_const_weighting_inner(tspace):
    """Test inner product with const weighting."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    constant = 1.5
    true_result_const = constant * np.vdot(yarr, xarr)

    w_const = odl.space_weighting(impl=tspace.impl, weight=constant)
    assert w_const.inner(x.data, y.data) == pytest.approx(true_result_const)

    # Exponent != 2 -> no inner
    w_const = odl.space_weighting(impl=tspace.impl, weight=constant, exponent=1)
    with pytest.raises(NotImplementedError):
        w_const.inner(x.data, y.data)


def test_const_weighting_norm(tspace, exponent):
    """Test norm with const weighting."""
    xarr, x = noise_elements(tspace)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)

    true_norm = factor * np.linalg.norm(xarr.ravel(), ord=exponent)

    w_const =  odl.space_weighting(impl=tspace.impl, weight=constant, exponent=exponent)


    if tspace.real_dtype == np.float16:
        tolerance = 5e-2
    elif tspace.real_dtype == np.float32:
        tolerance = 1e-6
    elif tspace.real_dtype == np.float64:
        tolerance = 1e-15
    elif tspace.real_dtype == np.float128:
        tolerance = 1e-19
    else:
        raise TypeError(f"No known tolerance for dtype {tspace.dtype}")
    
    assert w_const.norm(x) == pytest.approx(true_norm, rel=tolerance)


def test_const_weighting_dist(tspace, exponent):
    """Test dist with const weighting."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)


    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_dist = factor * np.linalg.norm((xarr - yarr).ravel(), ord=exponent)

    w_const = w_const = odl.space_weighting(impl=tspace.impl, weight=constant, exponent=exponent)

    if tspace.real_dtype == np.float16:
        tolerance = 5e-2
    elif tspace.real_dtype == np.float32:
        tolerance = 5e-7
    elif tspace.real_dtype == np.float64:
        tolerance = 1e-15
    elif tspace.real_dtype == np.float128:
        tolerance = 1e-19
    else:
        raise TypeError(f"No known tolerance for dtype {tspace.dtype}")

    assert w_const.dist(x, y) == pytest.approx(true_dist, rel=tolerance)



def test_custom_inner(tspace):
    """Test weighting with a custom inner product."""
    ns = tspace.array_namespace
    rtol = ns.sqrt(ns.finfo(tspace.dtype).resolution)

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    def inner(x, y):
        return ns.linalg.vecdot(y.ravel(), x.ravel())

    def dot(x,y):
        return ns.dot(x,y)
    
    w = odl.space_weighting(impl=tspace.impl, inner=inner)
    w_same = odl.space_weighting(impl=tspace.impl, inner=inner)
    w_other = odl.space_weighting(impl=tspace.impl, inner=dot)

    assert w == w
    assert w == w_same
    assert w != w_other

    true_inner = inner(xarr, yarr)
    assert w.inner(x.data, y.data) == pytest.approx(true_inner)

    true_norm = np.linalg.norm(xarr.ravel())
    assert w.norm(x.data) == pytest.approx(true_norm)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    assert w.dist(x.data, y.data) == pytest.approx(true_dist, rel=rtol)

    with pytest.raises(ValueError):
        odl.space_weighting(impl=tspace.impl, inner=inner, weight = 1)


def test_custom_norm(tspace):
    """Test weighting with a custom norm."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)
    ns = tspace.array_namespace

    def norm(x):
        return ns.linalg.norm(x)

    def other_norm(x):
        return ns.linalg.norm(x, ord=1)

    w = odl.space_weighting(impl=tspace.impl, norm=norm)
    w_same = odl.space_weighting(impl=tspace.impl, norm=norm)
    w_other = odl.space_weighting(impl=tspace.impl, norm=other_norm)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    true_norm = np.linalg.norm(xarr.ravel())
    assert tspace.norm(x) == pytest.approx(true_norm)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    assert tspace.dist(x, y) == pytest.approx(true_dist)

    with pytest.raises(ValueError):
        odl.space_weighting(impl=tspace.impl, norm=norm, weight = 1)


def test_custom_dist(tspace):
    """Test weighting with a custom dist."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)
    ns = tspace.array_namespace
    def dist(x, y):
        return ns.linalg.norm(x - y)

    def other_dist(x, y):
        return ns.linalg.norm(x - y, ord=1)

    w = odl.space_weighting(impl=tspace.impl, dist=dist)
    w_same = odl.space_weighting(impl=tspace.impl, dist=dist)
    w_other = odl.space_weighting(impl=tspace.impl, dist=other_dist)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    with pytest.raises(NotImplementedError):
        w.norm(x)

    true_dist = ns.linalg.norm((xarr - yarr).ravel())
    assert tspace.dist(x, y) == pytest.approx(true_dist)

    with pytest.raises(ValueError):
        odl.space_weighting(impl=tspace.impl, dist=dist, weight = 1)

def test_reduction(tspace):
    """Check that the generated docstrings are not empty."""
    ## In Pytorch 2.6, max and min reductions are not implemented for ComplexDouble dtype
    # <!> Can randomly raise RuntimeWarning: overflow encountered in reduce
    # <!> Can randomly raise AssertionError: assert (nan+8.12708086701316e-308j) == tensor(nan+8.1271e-308j, dtype=torch.complex128)
    x = tspace.element()
    backend = tspace.array_backend.array_namespace
    for name in ['sum', 'prod', 'min', 'max']:
        reduction = getattr(odl, name)
        reduction_arr = getattr(backend, name)
        if name in ['min', 'max'] and is_complex_dtype(tspace.dtype) and tspace.impl == 'pytorch':
            with pytest.raises(RuntimeError):
                assert reduction(x) == reduction_arr(x.data)
        else:
            assert reduction(x) == reduction_arr(x.data)


if __name__ == '__main__':
    odl.util.test_file(__file__)
    
