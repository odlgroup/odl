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
import pytest

import odl
from odl.set.space import LinearSpaceTypeError
from odl.space.entry_points import TENSOR_SPACE_IMPLS
from odl.space.npy_tensors import (
    NumpyTensor, NumpyTensorSpace)
from odl.util.testutils import (
    all_almost_equal, all_equal, noise_array, noise_element, noise_elements,
    isclose, simple_fixture)
from odl.core.array_API_support import lookup_array_backend
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
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_floating_dtype, 
        impl=impl, 
        device=device
        )

@pytest.fixture(scope='module', params=IMPL_DEVICE_PAIRS)
def floating_tspace(request, odl_floating_dtype):
    impl, device = request.param
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_floating_dtype, 
        impl=impl, 
        device=device
        )

@pytest.fixture(scope='module', params=IMPL_DEVICE_PAIRS)
def real_tspace(request, odl_real_floating_dtype):
    impl, device = request.param
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_real_floating_dtype, 
        impl=impl, 
        device=device
        )

@pytest.fixture(scope='module', params=IMPL_DEVICE_PAIRS)
def scalar_tspace(request, odl_scalar_dtype):
    impl, device = request.param
    return odl.tensor_space(
        shape=DEFAULT_SHAPE, 
        dtype=odl_scalar_dtype, 
        impl=impl, 
        device=device
        )

# --- Tests --- #
def test_init_tspace(floating_tspace):
    shape  = floating_tspace.shape
    impl   = floating_tspace.impl
    dtype  = floating_tspace.dtype
    device = floating_tspace.device

    # Weights
    constant_weighting = odl.space_weighting(
        impl,
        weight = 1.5
        )
    array_weighting    = odl.space_weighting(
        impl,
        device, 
        weight = _pos_array(odl.rn(
            shape, 
            impl=impl, dtype=dtype, device=device
            )
        ))
    
    tspace_impl = TENSOR_SPACE_IMPLS[impl]

    for weighting in [constant_weighting, array_weighting, None]:
        tspace_impl(
            DEFAULT_SHAPE, 
            dtype=dtype, 
            device=device, 
            weighting=weighting
            )

def test_properties(odl_impl_device_pairs):
    """Test that the space and element properties are as expected."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2,impl=impl, device=device)
    x = space.element()

    ns = space.array_namespace
    assert x.space is space
    assert x.ndim == space.ndim == 2
    assert x.dtype == space.dtype == getattr(ns, 'float32')
    assert x.size == space.size == 12
    assert x.shape == space.shape == DEFAULT_SHAPE
    assert x.itemsize == 4
    assert x.nbytes == 4 * 3 * 4
    assert x.device == device


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

def test_equals_space(tspace):
    """Test equality check of spaces."""
    impl = tspace.impl
    device = tspace.device
    dtype=tspace.dtype
    space = odl.tensor_space(3, impl=impl, dtype=dtype, device=device)
    same_space = odl.tensor_space(3, impl=impl, dtype=dtype, device=device)
    other_space = odl.tensor_space(4, impl=impl, dtype=dtype, device=device)

    assert space == space
    assert space == same_space
    assert space != other_space
    assert hash(space) == hash(same_space)
    assert hash(space) != hash(other_space)


def test_equals_elem(odl_impl_device_pairs):
    """Test equality check of space elements."""
    impl, device = odl_impl_device_pairs
    r3 = odl.rn(3, exponent=2, impl=impl, device=device)
    r3_1 = odl.rn(3, exponent=1, impl=impl, device=device)
    r4 = odl.rn(4, exponent=2, impl=impl, device=device)
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


def test_tspace_astype(odl_impl_device_pairs):
    """Test creation of a space counterpart with new dtype."""
    impl, device = odl_impl_device_pairs
    real_space = odl.rn(DEFAULT_SHAPE, impl=impl, device=device)
    int_space = odl.tensor_space(DEFAULT_SHAPE, dtype=int, impl=impl, device=device)
    assert real_space.astype(int) == int_space

    # Test propagation of weightings and the `[real/complex]_space` properties
    real = odl.rn(DEFAULT_SHAPE, weighting=1.5, impl=impl, device=device)
    cplx = odl.cn(DEFAULT_SHAPE, weighting=1.5, impl=impl, device=device)
    real_s = odl.rn(DEFAULT_SHAPE, weighting=1.5, dtype='float32', impl=impl, device=device)
    cplx_s = odl.cn(DEFAULT_SHAPE, weighting=1.5, dtype='complex64', impl=impl, device=device)

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
    # What the actual fuck
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


def test_lincomb_discontig(odl_impl_device_pairs):
    """Test lincomb with discontiguous input."""
    impl, device = odl_impl_device_pairs

    scalar_values = [0, 1, -1, 3.41]

    # Use small size for small array case
    tspace = odl.rn(DEFAULT_SHAPE, impl=impl, device=device)

    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b, discontig=True)

    # Use medium size to test fallback impls
    tspace = odl.rn((30, 40), impl=impl, device=device)

    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b, discontig=True)


def test_lincomb_exceptions(tspace):
    """Test whether lincomb raises correctly for bad output element."""
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
    ns = tspace.array_namespace
    y_pos_arr = ns.abs(y_arr) + 0.1

    # Testing standard positive integer power out-of-place and in-place
    assert all_almost_equal(x ** 2, x_arr ** 2)
    y **= 2
    y_arr **= 2
    assert all_almost_equal(y, y_arr)
    if tspace.impl == 'pytorch' and is_complex_dtype(tspace.dtype):
        pass
    else:
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
        ndigits = int(-math.log10(tspace.finfo().resolution) // 2)
    else:
        ndigits = int(-math.log10(tspace.finfo().resolution))

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
        ndigits = int(-math.log10(tspace.finfo().resolution) // 2)
    else:
        ndigits = int(-math.log10(tspace.finfo().resolution))

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
    ns = tspace.array_namespace
    # TODO: add weighting
    correct_inner = tspace.array_backend.to_cpu(
        ns.vdot(yarr.ravel(), xarr.ravel())
        )
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

    ns = tspace.array_namespace
    correct_norm = tspace.array_backend.to_cpu(
        ns.linalg.norm(xarr.ravel())
        )    

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


    correct_norm = ns.linalg.norm(xarr.ravel())

def test_norm_exceptions(tspace):
    """Test if norm raises correctly for bad input."""
    other_space = odl.rn((4, 3))
    other_x = other_space.zero()

    with pytest.raises(LinearSpaceTypeError):
        tspace.norm(other_x)


def test_pnorm(exponent, odl_impl_device_pairs):
    """Test the norm method with p!=2 against numpy.linalg.norm."""
    impl, device = odl_impl_device_pairs
    space_list = [
        odl.rn(DEFAULT_SHAPE, exponent=exponent,device=device, impl=impl),
        odl.cn(DEFAULT_SHAPE, exponent=exponent,device=device, impl=impl)
    ]
    for tspace in space_list:
        xarr, x = noise_elements(tspace)
        ns = tspace.array_namespace
        correct_norm = tspace.array_backend.to_cpu(ns.linalg.norm(xarr.ravel(), ord=exponent))

        assert tspace.norm(x) == pytest.approx(correct_norm)
        assert x.norm() == pytest.approx(correct_norm)


def test_dist(tspace):
    """Test the dist method against numpy.linalg.norm of the difference."""

    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    ns = tspace.array_namespace
    correct_dist = tspace.array_backend.to_cpu(
         ns.linalg.norm((xarr - yarr).ravel())
        )    

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


def test_pdist(odl_impl_device_pairs, exponent):
    """Test the dist method with p!=2 against numpy.linalg.norm of diff."""
    impl, device = odl_impl_device_pairs
    spaces = [
        odl.rn(DEFAULT_SHAPE, exponent=exponent, impl=impl, device=device),
        odl.cn(DEFAULT_SHAPE, exponent=exponent, impl=impl, device=device)
        ]
    # cls = odl.space.entry_points.tensor_space_impl(impl)

    # if complex in cls.available_dtypes:
    #     spaces.append(odl.cn(DEFAULT_SHAPE, exponent=exponent, impl=impl))

    for space in spaces:
        [xarr, yarr], [x, y] = noise_elements(space, n=2)
        ns = space.array_namespace
        correct_dist = space.array_backend.to_cpu(ns.linalg.norm((xarr - yarr).ravel(), ord=exponent))
        assert space.dist(x, y) == pytest.approx(correct_dist)
        assert x.dist(y) == pytest.approx(correct_dist)


def test_element_getitem(odl_impl_device_pairs, getitem_indices):
    """Check if getitem produces correct values, shape and other stuff."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl, device=device)
    x_arr, x = noise_elements(space)

    x_arr_sliced = x_arr[getitem_indices]
    sliced_shape = x_arr_sliced.shape
    x_sliced = x[getitem_indices]

    if x_arr_sliced.ndim == 0:
        try:
            assert x_arr_sliced == x_sliced
        except IndexError:
            assert x_arr_sliced[0] == x_sliced
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


def test_element_setitem(setitem_indices, odl_impl_device_pairs):
    """Check if setitem produces the same result as NumPy."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl, device=device)
    x_arr, x = noise_elements(space)



    x_arr_sliced = x_arr[setitem_indices]
    sliced_shape = x_arr_sliced.shape

    ns = space.array_namespace
    # Setting values with scalars
    x_arr[setitem_indices] = 2.3
    x[setitem_indices] = 2.3
    assert all_equal(x, x_arr)

    # Setting values with arrays
    rhs_arr = ns.ones(sliced_shape, device=device)
    x_arr[setitem_indices] = rhs_arr
    x[setitem_indices] = rhs_arr
    assert all_equal(x, x_arr)

    # Using a list of lists
    rhs_list = (-ns.ones(sliced_shape, device=device)).tolist()
    if impl != 'pytorch':
        x_arr[setitem_indices] = rhs_list
        x[setitem_indices] = rhs_list
        assert all_equal(x, x_arr)


def test_element_getitem_bool_array(odl_impl_device_pairs):
    """Check if getitem with boolean array yields the same result as NumPy."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl, device=device)
    bool_space = odl.tensor_space((2, 3, 4), dtype=bool, impl=impl, device=device)
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


def test_element_setitem_bool_array(odl_impl_device_pairs):
    """Check if setitem produces the same result as NumPy."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space((2, 3, 4), dtype='float32', exponent=1,
                             weighting=2, impl=impl, device=device)
    bool_space = odl.tensor_space((2, 3, 4), dtype=bool, impl=impl, device=device)
    x_arr, x = noise_elements(space)
    cond_arr, cond = noise_elements(bool_space)
    ns = space.array_namespace

    x_arr_sliced = x_arr[cond_arr]
    sliced_shape = x_arr_sliced.shape

    # Setting values with scalars
    x_arr[cond_arr] = 2.3
    x[cond] = 2.3
    assert all_equal(x, x_arr)

    # Setting values with arrays
    rhs_arr = ns.ones(sliced_shape, device=device)
    x_arr[cond_arr] = rhs_arr
    x[cond] = rhs_arr
    assert all_equal(x, x_arr)

    # Using a list of lists
    rhs_list = (-ns.ones(sliced_shape, device=device)).tolist()
    if impl == 'pytorch':
        cond_arr = bool_space.array_backend.array_constructor(cond_arr, device=device)
        rhs_list = bool_space.array_backend.array_constructor(rhs_list, device=device)
    else:
        x_arr[cond_arr] = rhs_list
    x[cond] = rhs_list
    assert all_equal(x, x_arr)


def test_transpose(odl_impl_device_pairs):
    """Test the .T property of tensors against plain inner product."""
    impl, device = odl_impl_device_pairs
    spaces = [
        odl.rn(DEFAULT_SHAPE, impl=impl, device=device),
        odl.cn(DEFAULT_SHAPE, impl=impl, device=device)
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
    output = x * 1.0
    assert x * 1.0 in tspace
    assert 1.0 * x in tspace


def test_member_copy(odl_impl_device_pairs):
    """Test copy method of elements."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2, impl=impl, device = device)
    x = noise_element(space)

    y = x.copy()
    assert x == y
    assert y is not x

    # Check that result is not aliased
    x *= 2
    assert x != y


def test_python_copy(odl_impl_device_pairs):
    """Test compatibility with the Python copy module."""
    import copy
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2, impl=impl, device = device)
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

def test_conversion_to_scalar(odl_impl_device_pairs):
    """Test conversion of size-1 vectors/tensors to scalars."""
    impl, device = odl_impl_device_pairs
    space = odl.rn(1, impl=impl, device=device)
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

    with pytest.raises(AssertionError):
        int(element)
    with pytest.raises(AssertionError):
        float(element)
    with pytest.raises(AssertionError):
        complex(element)

def test_bool_conversion(odl_impl_device_pairs):
    """Verify that the __bool__ function works."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space(2, dtype='float32', impl=impl, device = device)
    x = space.element([0, 1])

    with pytest.raises(ValueError):
        bool(x)
    assert odl.any(x)
    assert any(x)
    assert not odl.all(x)
    assert not all(x)

    space = odl.tensor_space(1, dtype='float32', impl=impl, device = device)
    x = space.one()

    assert odl.any(x)
    assert any(x)
    assert odl.all(x)
    assert all(x)

def test_array_wrap_method(odl_impl_device_pairs):
    """Verify that the __array_wrap__ method for NumPy works."""
    impl, device = odl_impl_device_pairs
    space = odl.tensor_space(DEFAULT_SHAPE, dtype='float32', exponent=1, weighting=2,impl=impl, device=device)
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


def test_array_weighting_array_is_valid(odl_impl_device_pairs):
    """Test the is_valid method of array weightings."""
    impl, device = odl_impl_device_pairs
    space = odl.rn(DEFAULT_SHAPE, impl=impl, device=device)
    weight_arr = _pos_array(space)

    assert odl.space_weighting(impl, weight=weight_arr, device=device)
    # Invalid
    weight_arr[0] = 0
    with pytest.raises(ValueError):
        odl.space_weighting(impl, weight=weight_arr, device=device)


def test_array_weighting_equals(odl_impl_device_pairs):
    """Test the equality check method of array weightings."""
    impl, device = odl_impl_device_pairs
    space = odl.rn(5, impl=impl, device=device)
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)

    weighting_arr = odl.space_weighting(impl, weight=weight_arr, device=device)
    weighting_arr2 = odl.space_weighting(impl, weight=weight_arr, device=device)
    weighting_elem = odl.space_weighting(impl, weight=weight_elem, device=device)
    weighting_elem_copy = odl.space_weighting(impl, weight=weight_elem.copy(), device=device)
    weighting_elem2 = odl.space_weighting(impl, weight=weight_elem, device=device)
    weighting_other_arr = odl.space_weighting(impl, weight=weight_arr +1 , device=device)
    weighting_other_exp = odl.space_weighting(impl, weight=weight_arr +1, exponent=1, device=device)

    assert weighting_arr == weighting_arr2
    assert weighting_arr == weighting_elem
    assert weighting_arr == weighting_elem_copy
    assert weighting_elem == weighting_elem2
    assert weighting_arr != weighting_other_arr
    assert weighting_arr != weighting_other_exp


def test_array_weighting_equiv(odl_impl_device_pairs):
    """Test the equiv method of Numpy array weightings."""
    impl, device = odl_impl_device_pairs
    space = odl.rn(5, impl=impl, device=device)
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)
    different_arr = weight_arr + 1
    w_arr = odl.space_weighting(impl, weight=weight_arr, device=device)
    w_elem = odl.space_weighting(impl, weight=weight_elem, device=device)
    w_different_arr = odl.space_weighting(impl, weight=different_arr, device=device)

    ns = space.array_namespace

    # Equal -> True
    assert w_arr.equiv(w_arr)
    assert w_arr.equiv(w_elem)
    # Different array -> False
    assert not w_arr.equiv(w_different_arr)

    # Test shortcuts in the implementation
    const_arr = ns.ones(space.shape, device=device) * 1.5
    w_const_arr = odl.space_weighting(impl, weight=const_arr, device=device)
    w_const = odl.space_weighting(impl, weight=1.5, device=device)
    w_wrong_const = odl.space_weighting(impl, weight=1, device=device)
    w_wrong_exp = odl.space_weighting(impl, weight=1.5, exponent=1, device=device)

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
    weighting = odl.space_weighting(
        impl = tspace.impl, 
        weight = weight_arr,
        device = tspace.device
        )
    
    ns = tspace.array_namespace

    true_inner = ns.vdot(yarr.ravel(), (xarr * weight_arr).ravel())
    assert weighting.inner(x.data, y.data) == pytest.approx(tspace.array_backend.to_cpu(true_inner))

    # Exponent != 2 -> no inner product, should raise
    with pytest.raises(NotImplementedError):
        odl.space_weighting(impl = tspace.impl, weight =weight_arr, exponent=1.0, device = tspace.device).inner(x.data, y.data)


def test_array_weighting_norm(tspace, exponent):
    """Test norm in a weighted space."""
    ns = tspace.array_namespace
    rtol = math.sqrt(ns.finfo(tspace.dtype).resolution)
    xarr, x = noise_elements(tspace)

    weight_arr = _pos_array(tspace)
    weighting = odl.space_weighting(impl = tspace.impl, weight=weight_arr, exponent=exponent, device =tspace.device)

    if exponent == float('inf'):
        true_norm = ns.linalg.vector_norm(
            weight_arr * xarr,
            ord=exponent)
    else:
        true_norm = ns.linalg.norm(
            (weight_arr ** (1 / exponent) * xarr).ravel(),
            ord=exponent)

    assert weighting.norm(x.data) == pytest.approx(
        tspace.array_backend.to_cpu(true_norm), rel=rtol)


def test_array_weighting_dist(tspace, exponent):
    """Test dist product in a weighted space."""
    ns = tspace.array_namespace
    rtol = math.sqrt(ns.finfo(tspace.dtype).resolution)
    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    weight_arr = _pos_array(tspace)
    weighting = odl.space_weighting(impl = tspace.impl, weight=weight_arr, exponent=exponent, device=tspace.device)

    if exponent == float('inf'):
        true_dist = ns.linalg.norm(
            (weight_arr * (xarr - yarr)).ravel(),
            ord=float('inf'))
    else:
        true_dist = ns.linalg.norm(
            (weight_arr ** (1 / exponent) * (xarr - yarr)).ravel(),
            ord=exponent)

    assert weighting.dist(x.data, y.data) == pytest.approx(
        tspace.array_backend.to_cpu(true_dist), rel=rtol)


def test_const_weighting_init(odl_impl_device_pairs, exponent):
    """Test initialization of constant weightings."""
    impl, device = odl_impl_device_pairs
    # Just test if the code runs
    odl.space_weighting(impl=impl, weight=1.5, exponent=exponent, device=device)
    with pytest.raises(ValueError):
        odl.space_weighting(impl=impl, weight=0, exponent=exponent, device=device)
    with pytest.raises(ValueError):
        odl.space_weighting(impl=impl, weight=-1.5, exponent=exponent, device=device)
    with pytest.raises(ValueError):
        odl.space_weighting(impl=impl, weight=float('inf'), exponent=exponent, device=device)


def test_const_weighting_comparison(tspace):
    """Test equality to and equivalence with const weightings."""
    odl_tspace_impl = tspace.impl
    ns = tspace.array_namespace
    constant = 1.5

    w_const = odl.space_weighting(impl=odl_tspace_impl, weight=constant)
    w_const2 = odl.space_weighting(impl=odl_tspace_impl, weight=constant)
    w_other_const = odl.space_weighting(impl=odl_tspace_impl, weight=constant+1)
    w_other_exp = odl.space_weighting(impl=odl_tspace_impl, weight=constant, exponent = 1)

    const_arr = constant * ns.ones(DEFAULT_SHAPE)

    w_const_arr = odl.space_weighting(impl=odl_tspace_impl, weight=const_arr)
    other_const_arr = (constant + 1) * ns.ones(DEFAULT_SHAPE)
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

    ns = tspace.array_namespace

    constant = 1.5
    true_result_const = constant * ns.vecdot(yarr.ravel(), xarr.ravel())

    w_const = odl.space_weighting(impl=tspace.impl, weight=constant)

    assert w_const.inner(x, y) == true_result_const

    # Exponent != 2 -> no inner
    w_const = odl.space_weighting(impl=tspace.impl, weight=constant, exponent=1)
    with pytest.raises(NotImplementedError):
        w_const.inner(x, y)


def test_const_weighting_norm(tspace, exponent):
    """Test norm with const weighting."""
    xarr, x = noise_elements(tspace)

    ns = tspace.array_namespace

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)

    true_norm = float(factor * ns.linalg.norm(xarr.ravel(), ord=exponent))

    w_const =  odl.space_weighting(impl=tspace.impl, weight=constant, exponent=exponent)

    array_backend = tspace.array_backend
    real_dtype = array_backend.identifier_of_dtype(tspace.real_dtype)

    if real_dtype == "float16":
        tolerance = 5e-2
    elif real_dtype == "float32":
        tolerance = 5e-6
    elif real_dtype == "float64" or real_dtype == float:
        tolerance = 1e-15
    elif real_dtype == "float128":
        tolerance = 1e-19
    else:
        raise TypeError(f"No known tolerance for dtype {real_dtype}")
    
    # assert w_const.norm(x) == pytest.approx(true_norm, rel=tolerance)
    assert isclose(w_const.norm(x), true_norm, rtol=tolerance)


def test_const_weighting_dist(tspace, exponent):
    """Test dist with const weighting."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    ns = tspace.array_namespace

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_dist = float(factor * ns.linalg.norm((xarr - yarr).ravel(), ord=exponent))
    w_const = w_const = odl.space_weighting(impl=tspace.impl, weight=constant, exponent=exponent)

    array_backend = tspace.array_backend
    real_dtype = array_backend.identifier_of_dtype(tspace.real_dtype)
    if real_dtype == "float16":
        tolerance = 5e-2
    elif real_dtype == "float32":
        tolerance = 5e-7
    elif real_dtype == "float64" or real_dtype == float:
        tolerance = 1e-15
    elif real_dtype == "float128":
        tolerance = 1e-19
    else:
        raise TypeError(f"No known tolerance for dtype {real_dtype}")

    # assert w_const.dist(x, y) == pytest.approx(true_dist, rel=tolerance)
    assert isclose(w_const.dist(x,y), true_dist, rtol=tolerance)



def test_custom_inner(tspace):
    """Test weighting with a custom inner product."""
    ns = tspace.array_namespace
    rtol = math.sqrt(ns.finfo(tspace.dtype).resolution)

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    def inner(x, y):
        return ns.linalg.vecdot(y.ravel(), x.ravel())
    
    def inner_lspacelement(x, y):
        return ns.linalg.vecdot(y.data.ravel(), x.data.ravel())

    def dot(x,y):
        return ns.dot(x,y)
    
    w = odl.space_weighting(impl=tspace.impl, inner=inner_lspacelement)
    w_same = odl.space_weighting(impl=tspace.impl, inner=inner_lspacelement)
    w_other = odl.space_weighting(impl=tspace.impl, inner=dot)

    assert w == w
    assert w == w_same
    assert w != w_other

    true_inner = inner(xarr, yarr)
    assert isclose(w.inner(x, y), true_inner)

    true_norm = float(ns.linalg.norm(xarr.ravel()))
    assert isclose(w.norm(x), true_norm)

    true_dist = float(ns.linalg.norm((xarr - yarr).ravel()))
    assert isclose( w.dist(x, y), true_dist, rtol=rtol)

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

    true_norm = ns.linalg.norm(xarr.ravel())
    pytest.approx(tspace.norm(x), true_norm)

    true_dist = ns.linalg.norm((xarr - yarr).ravel())
    pytest.approx(tspace.dist(x, y), true_dist)

    with pytest.raises(ValueError):
        odl.space_weighting(impl=tspace.impl, norm=norm, weight = 1)


def test_custom_dist(tspace):
    """Test weighting with a custom dist."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)
    ns = tspace.array_namespace
    def dist(x, y):
        return ns.linalg.norm(x - y)
    
    def dist_lspace_element(x, y):
        return ns.linalg.norm(x.data - y.data)

    def other_dist(x, y):
        return ns.linalg.norm(x - y, ord=1)

    w = odl.space_weighting(impl=tspace.impl, dist=dist_lspace_element)
    w_same = odl.space_weighting(impl=tspace.impl, dist=dist_lspace_element)
    w_other = odl.space_weighting(impl=tspace.impl, dist=other_dist)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    with pytest.raises(NotImplementedError):
        w.norm(x)

    true_dist = ns.linalg.norm((xarr - yarr).ravel())
    pytest.approx(tspace.dist(x, y), true_dist)

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
    
