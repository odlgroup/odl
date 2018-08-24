# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for Numpy-based tensors."""

from __future__ import division
import numpy as np
import operator
from packaging.version import parse as parse_version
import pytest
import sys

import odl
from odl.set.space import LinearSpaceTypeError
from odl.space.npy_tensors import (
    NumpyTensor, NumpyTensorSpace,
    NumpyTensorSpaceConstWeighting, NumpyTensorSpaceArrayWeighting,
    NumpyTensorSpaceCustomInner, NumpyTensorSpaceCustomNorm,
    NumpyTensorSpaceCustomDist)
from odl.util.testutils import (
    all_almost_equal, all_equal, simple_fixture,
    noise_array, noise_element, noise_elements)
from odl.util.ufuncs import UFUNCS


# --- Test helpers --- #

PYTHON2 = sys.version_info.major < 3
USE_ARRAY_UFUNCS_INTERFACE = (parse_version(np.__version__) >=
                              parse_version('1.13'))


# Functions to return arrays and classes corresponding to impls. Extend
# when a new impl is available.

def _pos_array(space):
    """Create an array with positive real entries in ``space``."""
    return np.abs(noise_array(space)) + 0.1


def _array_cls(impl):
    """Return the array class for given impl."""
    if impl == 'numpy':
        return np.ndarray
    else:
        assert False


def _odl_tensor_cls(impl):
    """Return the ODL tensor class for given impl."""
    if impl == 'numpy':
        return NumpyTensor
    else:
        assert False


def _weighting_cls(impl, kind):
    """Return the weighting class for given impl and kind."""
    if impl == 'numpy':
        if kind == 'array':
            return NumpyTensorSpaceArrayWeighting
        elif kind == 'const':
            return NumpyTensorSpaceConstWeighting
        elif kind == 'inner':
            return NumpyTensorSpaceCustomInner
        elif kind == 'norm':
            return NumpyTensorSpaceCustomNorm
        elif kind == 'dist':
            return NumpyTensorSpaceCustomDist
        else:
            assert False
    else:
        assert False


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


@pytest.fixture(scope='module', params=weight_params, ids=weight_ids)
def weight(request):
    return request.param


@pytest.fixture(scope='module')
def tspace(odl_floating_dtype, odl_tspace_impl):
    impl = odl_tspace_impl
    dtype = odl_floating_dtype
    return odl.tensor_space(shape=(3, 4), dtype=dtype, impl=impl)


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

    if impl == 'numpy':
        if isinstance(weight, np.ndarray):
            weighting_cls = _weighting_cls(impl, 'array')
        else:
            weighting_cls = _weighting_cls(impl, 'const')
    else:
        assert False

    weighting = weighting_cls(weight, exponent)

    assert space.weighting == weighting

    # Using a weighting instance
    space = odl.tensor_space((3, 4), weighting=weighting, exponent=exponent,
                             impl=impl)
    assert space.weighting is weighting

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
    assert elem.shape == elem.data.shape
    assert elem.dtype == tspace.dtype == elem.data.dtype
    if order is not None:
        assert elem.data.flags[order + '_CONTIGUOUS']

    # From space elements
    other_elem = tspace.element(np.ones(tspace.shape))
    elem = tspace.element(other_elem, order=order)
    assert all_equal(elem, other_elem)
    if order is None:
        assert elem is other_elem
    else:
        assert elem.data.flags[order + '_CONTIGUOUS']

    # From Numpy array (C order)
    arr_c = np.random.rand(*tspace.shape).astype(tspace.dtype)
    elem = tspace.element(arr_c, order=order)
    assert all_equal(elem, arr_c)
    assert elem.shape == elem.data.shape
    assert elem.dtype == tspace.dtype == elem.data.dtype
    if order is None or order == 'C':
        # None or same order should not lead to copy
        assert np.may_share_memory(elem.data, arr_c)
    if order is not None:
        # Contiguousness in explicitly provided order should be guaranteed
        assert elem.data.flags[order + '_CONTIGUOUS']

    # From Numpy array (F order)
    arr_f = np.asfortranarray(arr_c)
    elem = tspace.element(arr_f, order=order)
    assert all_equal(elem, arr_f)
    assert elem.shape == elem.data.shape
    assert elem.dtype == tspace.dtype == elem.data.dtype
    if order is None or order == 'F':
        # None or same order should not lead to copy
        assert np.may_share_memory(elem.data, arr_f)
    if order is not None:
        # Contiguousness in explicitly provided order should be guaranteed
        assert elem.data.flags[order + '_CONTIGUOUS']

    # From pointer
    arr_c_ptr = arr_c.ctypes.data
    elem = tspace.element(data_ptr=arr_c_ptr, order='C')
    assert all_equal(elem, arr_c)
    assert np.may_share_memory(elem.data, arr_c)
    arr_f_ptr = arr_f.ctypes.data
    elem = tspace.element(data_ptr=arr_f_ptr, order='F')
    assert all_equal(elem, arr_f)
    assert np.may_share_memory(elem.data, arr_f)

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
    xd = noise_element(tspace)
    yd = noise_element(tspace)

    # TODO: add weighting
    correct_inner = np.vdot(yd, xd)
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

    correct_norm = np.linalg.norm(xarr.ravel())
    assert tspace.norm(x) == pytest.approx(correct_norm)
    assert x.norm() == pytest.approx(correct_norm)


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
        correct_norm = np.linalg.norm(xarr.ravel(), ord=exponent)

        assert tspace.norm(x) == pytest.approx(correct_norm)
        assert x.norm() == pytest.approx(correct_norm)


def test_dist(tspace):
    """Test the dist method against numpy.linalg.norm of the difference."""
    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    correct_dist = np.linalg.norm((xarr - yarr).ravel())
    assert tspace.dist(x, y) == pytest.approx(correct_dist)
    assert x.dist(y) == pytest.approx(correct_dist)


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
    spaces = [odl.rn((3, 4), impl=impl)]
    cls = odl.space.entry_points.tensor_space_impl(impl)
    if complex in cls.available_dtypes():
        spaces.append(odl.cn((3, 4), impl=impl))

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


def test_multiply_by_scalar(tspace):
    """Verify that mult. with NumPy scalars preserves the element type."""
    x = tspace.zero()
    assert x * 1.0 in tspace
    assert x * np.float32(1.0) in tspace
    assert 1.0 * x in tspace
    assert np.float32(1.0) * x in tspace


def test_member_copy(odl_tspace_impl):
    """Test copy method of elements."""
    impl = odl_tspace_impl
    space = odl.tensor_space((3, 4), dtype='float32', exponent=1, weighting=2,
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
    space = odl.tensor_space((3, 4), dtype='float32', exponent=1, weighting=2,
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
    if PYTHON2:
        assert long(element) == long(value)

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

    with pytest.raises(TypeError):
        int(element)
    with pytest.raises(TypeError):
        float(element)
    with pytest.raises(TypeError):
        complex(element)
    if PYTHON2:
        with pytest.raises(TypeError):
            long(element)


def test_bool_conversion(odl_tspace_impl):
    """Verify that the __bool__ function works."""
    impl = odl_tspace_impl
    space = odl.tensor_space(2, dtype='float32', impl=impl)
    x = space.element([0, 1])

    with pytest.raises(ValueError):
        bool(x)
    assert np.any(x)
    assert any(x)
    assert not np.all(x)
    assert not all(x)

    space = odl.tensor_space(1, dtype='float32', impl=impl)
    x = space.one()

    assert np.any(x)
    assert any(x)
    assert np.all(x)
    assert all(x)


def test_numpy_array_interface(odl_tspace_impl):
    """Verify that the __array__ interface for NumPy works."""
    impl = odl_tspace_impl
    space = odl.tensor_space((3, 4), dtype='float32', exponent=1, weighting=2,
                             impl=impl)
    x = space.one()
    arr = x.__array__()

    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.ones(x.shape))

    x_arr = np.array(x)
    assert np.array_equal(x_arr, np.ones(x.shape))
    x_as_arr = np.asarray(x)
    assert np.array_equal(x_as_arr, np.ones(x.shape))
    x_as_any_arr = np.asanyarray(x)
    assert np.array_equal(x_as_any_arr, np.ones(x.shape))


def test_array_wrap_method(odl_tspace_impl):
    """Verify that the __array_wrap__ method for NumPy works."""
    impl = odl_tspace_impl
    space = odl.tensor_space((3, 4), dtype='float32', exponent=1, weighting=2,
                             impl=impl)
    x_arr, x = noise_elements(space)
    y_arr = np.sin(x_arr)
    y = np.sin(x)  # Should yield again an ODL tensor

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


def test_array_weighting_init(odl_tspace_impl, exponent):
    """Test initialization of array weightings."""
    impl = odl_tspace_impl
    space = odl.rn((3, 4), impl=impl)
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)

    weighting_cls = _weighting_cls(impl, 'array')
    weighting_arr = weighting_cls(weight_arr, exponent=exponent)
    weighting_elem = weighting_cls(weight_elem, exponent=exponent)

    assert isinstance(weighting_arr.array, _array_cls(impl))
    assert isinstance(weighting_elem.array, _array_cls(impl))


def test_array_weighting_array_is_valid(odl_tspace_impl):
    """Test the is_valid method of array weightings."""
    impl = odl_tspace_impl
    space = odl.rn((3, 4), impl=impl)
    weight_arr = _pos_array(space)

    weighting_cls = _weighting_cls(impl, 'array')
    weighting_arr = weighting_cls(weight_arr)

    assert weighting_arr.is_valid()

    # Invalid
    weight_arr[0] = 0
    weighting_arr = NumpyTensorSpaceArrayWeighting(weight_arr)
    assert not weighting_arr.is_valid()


def test_array_weighting_equals(odl_tspace_impl):
    """Test the equality check method of array weightings."""
    impl = odl_tspace_impl
    space = odl.rn(5, impl=impl)
    weight_arr = _pos_array(space)
    weight_elem = space.element(weight_arr)

    weighting_cls = _weighting_cls(impl, 'array')
    weighting_arr = weighting_cls(weight_arr)
    weighting_arr2 = weighting_cls(weight_arr)
    weighting_elem = weighting_cls(weight_elem)
    weighting_elem_copy = weighting_cls(weight_elem.copy())
    weighting_elem2 = weighting_cls(weight_elem)
    weighting_other_arr = weighting_cls(weight_arr - 1)
    weighting_other_exp = weighting_cls(weight_arr - 1, exponent=1)

    assert weighting_arr == weighting_arr2
    assert weighting_arr == weighting_elem
    assert weighting_arr != weighting_elem_copy
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

    arr_weighting_cls = _weighting_cls(impl, 'array')
    w_arr = arr_weighting_cls(weight_arr)
    w_elem = arr_weighting_cls(weight_elem)
    w_different_arr = arr_weighting_cls(different_arr)

    # Equal -> True
    assert w_arr.equiv(w_arr)
    assert w_arr.equiv(w_elem)
    # Different array -> False
    assert not w_arr.equiv(w_different_arr)

    # Test shortcuts in the implementation
    const_arr = np.ones(space.shape) * 1.5

    const_weighting_cls = _weighting_cls(impl, 'const')
    w_const_arr = arr_weighting_cls(const_arr)
    w_const = const_weighting_cls(1.5)
    w_wrong_const = const_weighting_cls(1)
    w_wrong_exp = const_weighting_cls(1.5, exponent=1)

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
    weighting = NumpyTensorSpaceArrayWeighting(weight_arr)

    true_inner = np.vdot(yarr, xarr * weight_arr)
    assert weighting.inner(x, y) == pytest.approx(true_inner)

    # Exponent != 2 -> no inner product, should raise
    with pytest.raises(NotImplementedError):
        NumpyTensorSpaceArrayWeighting(weight_arr, exponent=1.0).inner(x, y)


def test_array_weighting_norm(tspace, exponent):
    """Test norm in a weighted space."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)
    xarr, x = noise_elements(tspace)

    weight_arr = _pos_array(tspace)
    weighting = NumpyTensorSpaceArrayWeighting(weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_norm = np.linalg.norm(
            (weight_arr * xarr).ravel(),
            ord=float('inf'))
    else:
        true_norm = np.linalg.norm(
            (weight_arr ** (1 / exponent) * xarr).ravel(),
            ord=exponent)

    assert weighting.norm(x) == pytest.approx(true_norm, rel=rtol)


def test_array_weighting_dist(tspace, exponent):
    """Test dist product in a weighted space."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)
    [xarr, yarr], [x, y] = noise_elements(tspace, n=2)

    weight_arr = _pos_array(tspace)
    weighting = NumpyTensorSpaceArrayWeighting(weight_arr, exponent=exponent)

    if exponent == float('inf'):
        true_dist = np.linalg.norm(
            (weight_arr * (xarr - yarr)).ravel(),
            ord=float('inf'))
    else:
        true_dist = np.linalg.norm(
            (weight_arr ** (1 / exponent) * (xarr - yarr)).ravel(),
            ord=exponent)

    assert weighting.dist(x, y) == pytest.approx(true_dist, rel=rtol)


def test_const_weighting_init(odl_tspace_impl, exponent):
    """Test initialization of constant weightings."""
    impl = odl_tspace_impl
    constant = 1.5

    # Just test if the code runs
    weighting_cls = _weighting_cls(impl, 'const')
    weighting_cls(constant, exponent=exponent)

    with pytest.raises(ValueError):
        weighting_cls(0)
    with pytest.raises(ValueError):
        weighting_cls(-1)
    with pytest.raises(ValueError):
        weighting_cls(float('inf'))


def test_const_weighting_comparison(odl_tspace_impl):
    """Test equality to and equivalence with const weightings."""
    impl = odl_tspace_impl
    constant = 1.5

    const_weighting_cls = _weighting_cls(impl, 'const')
    w_const = const_weighting_cls(constant)
    w_const2 = const_weighting_cls(constant)
    w_other_const = const_weighting_cls(constant + 1)
    w_other_exp = const_weighting_cls(constant, exponent=1)

    const_arr = constant * np.ones((3, 4))

    arr_weighting_cls = _weighting_cls(impl, 'array')
    w_const_arr = arr_weighting_cls(const_arr)
    other_const_arr = (constant + 1) * np.ones((3, 4))
    w_other_const_arr = arr_weighting_cls(other_const_arr)

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

    w_const = NumpyTensorSpaceConstWeighting(constant)
    assert w_const.inner(x, y) == pytest.approx(true_result_const)

    # Exponent != 2 -> no inner
    w_const = NumpyTensorSpaceConstWeighting(constant, exponent=1)
    with pytest.raises(NotImplementedError):
        w_const.inner(x, y)


def test_const_weighting_norm(tspace, exponent):
    """Test norm with const weighting."""
    xarr, x = noise_elements(tspace)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_norm = factor * np.linalg.norm(xarr.ravel(), ord=exponent)

    w_const = NumpyTensorSpaceConstWeighting(constant, exponent=exponent)
    assert w_const.norm(x) == pytest.approx(true_norm)


def test_const_weighting_dist(tspace, exponent):
    """Test dist with const weighting."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    constant = 1.5
    if exponent == float('inf'):
        factor = constant
    else:
        factor = constant ** (1 / exponent)
    true_dist = factor * np.linalg.norm((xarr - yarr).ravel(), ord=exponent)

    w_const = NumpyTensorSpaceConstWeighting(constant, exponent=exponent)
    assert w_const.dist(x, y) == pytest.approx(true_dist)


def test_custom_inner(tspace):
    """Test weighting with a custom inner product."""
    rtol = np.sqrt(np.finfo(tspace.dtype).resolution)

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    def inner(x, y):
        return np.vdot(y, x)

    w = NumpyTensorSpaceCustomInner(inner)
    w_same = NumpyTensorSpaceCustomInner(inner)
    w_other = NumpyTensorSpaceCustomInner(np.dot)

    assert w == w
    assert w == w_same
    assert w != w_other

    true_inner = inner(xarr, yarr)
    assert w.inner(x, y) == pytest.approx(true_inner)

    true_norm = np.linalg.norm(xarr.ravel())
    assert w.norm(x) == pytest.approx(true_norm)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    assert w.dist(x, y) == pytest.approx(true_dist, rel=rtol)

    with pytest.raises(TypeError):
        NumpyTensorSpaceCustomInner(1)


def test_custom_norm(tspace):
    """Test weighting with a custom norm."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    norm = np.linalg.norm

    def other_norm(x):
        return np.linalg.norm(x, ord=1)

    w = NumpyTensorSpaceCustomNorm(norm)
    w_same = NumpyTensorSpaceCustomNorm(norm)
    w_other = NumpyTensorSpaceCustomNorm(other_norm)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    true_norm = np.linalg.norm(xarr.ravel())
    assert w.norm(x) == pytest.approx(true_norm)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    assert w.dist(x, y) == pytest.approx(true_dist)

    with pytest.raises(TypeError):
        NumpyTensorSpaceCustomNorm(1)


def test_custom_dist(tspace):
    """Test weighting with a custom dist."""
    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    def dist(x, y):
        return np.linalg.norm(x - y)

    def other_dist(x, y):
        return np.linalg.norm(x - y, ord=1)

    w = NumpyTensorSpaceCustomDist(dist)
    w_same = NumpyTensorSpaceCustomDist(dist)
    w_other = NumpyTensorSpaceCustomDist(other_dist)

    assert w == w
    assert w == w_same
    assert w != w_other

    with pytest.raises(NotImplementedError):
        w.inner(x, y)

    with pytest.raises(NotImplementedError):
        w.norm(x)

    true_dist = np.linalg.norm((xarr - yarr).ravel())
    assert w.dist(x, y) == pytest.approx(true_dist)

    with pytest.raises(TypeError):
        NumpyTensorSpaceCustomDist(1)


# --- Ufuncs & Reductions --- #


def test_ufuncs(tspace, odl_ufunc):
    """Test ufuncs in ``x.ufuncs`` against direct Numpy ufuncs."""
    name = odl_ufunc

    # Get the ufunc from numpy as reference, plus some additional info
    npy_ufunc = getattr(np, name)
    nin = npy_ufunc.nin
    nout = npy_ufunc.nout

    if (np.issubsctype(tspace.dtype, np.floating) or
            np.issubsctype(tspace.dtype, np.complexfloating) and
            name in ['bitwise_and',
                     'bitwise_or',
                     'bitwise_xor',
                     'invert',
                     'left_shift',
                     'right_shift']):
        # Skip integer only methods for floating point data types
        return

    if (np.issubsctype(tspace.dtype, np.complexfloating) and
            name in ['remainder',
                     'trunc',
                     'signbit',
                     'invert',
                     'left_shift',
                     'right_shift',
                     'rad2deg',
                     'deg2rad',
                     'copysign',
                     'mod',
                     'modf',
                     'fmod',
                     'logaddexp2',
                     'logaddexp',
                     'hypot',
                     'arctan2',
                     'floor',
                     'ceil']):
        # Skip real-only methods for complex data types
        return

    # Create some data
    arrays, elements = noise_elements(tspace, nin + nout)
    in_arrays = arrays[:nin]
    out_arrays = arrays[nin:]
    data_elem = elements[0]

    out_elems = elements[nin:]
    if nout == 1:
        out_arr_kwargs = {'out': out_arrays[0]}
        out_elem_kwargs = {'out': out_elems[0]}
    elif nout > 1:
        out_arr_kwargs = {'out': out_arrays[:nout]}
        out_elem_kwargs = {'out': out_elems[:nout]}

    # Get function to call, using both interfaces:
    # - vec.ufunc(other_args)
    # - np.ufunc(vec, other_args)
    elem_fun_old = getattr(data_elem.ufuncs, name)
    in_elems_old = elements[1:nin]
    elem_fun_new = npy_ufunc
    in_elems_new = elements[:nin]

    # Out-of-place
    npy_result = npy_ufunc(*in_arrays)
    odl_result_old = elem_fun_old(*in_elems_old)
    assert all_almost_equal(npy_result, odl_result_old)
    odl_result_new = elem_fun_new(*in_elems_new)
    assert all_almost_equal(npy_result, odl_result_new)

    # Test type of output
    if nout == 1:
        assert isinstance(odl_result_old, tspace.element_type)
        assert isinstance(odl_result_new, tspace.element_type)
    elif nout > 1:
        for i in range(nout):
            assert isinstance(odl_result_old[i], tspace.element_type)
            assert isinstance(odl_result_new[i], tspace.element_type)

    # In-place with ODL objects as `out`
    npy_result = npy_ufunc(*in_arrays, **out_arr_kwargs)
    odl_result_old = elem_fun_old(*in_elems_old, **out_elem_kwargs)
    assert all_almost_equal(npy_result, odl_result_old)
    if USE_ARRAY_UFUNCS_INTERFACE:
        # In-place will not work with Numpy < 1.13
        odl_result_new = elem_fun_new(*in_elems_new, **out_elem_kwargs)
        assert all_almost_equal(npy_result, odl_result_new)

    # Check that returned stuff refers to given out
    if nout == 1:
        assert odl_result_old is out_elems[0]
        if USE_ARRAY_UFUNCS_INTERFACE:
            assert odl_result_new is out_elems[0]
    elif nout > 1:
        for i in range(nout):
            assert odl_result_old[i] is out_elems[i]
            if USE_ARRAY_UFUNCS_INTERFACE:
                assert odl_result_new[i] is out_elems[i]

    # In-place with Numpy array as `out` for new interface
    if USE_ARRAY_UFUNCS_INTERFACE:
        out_arrays_new = [np.empty_like(arr) for arr in out_arrays]
        if nout == 1:
            out_elem_kwargs_new = {'out': out_arrays_new[0]}
        elif nout > 1:
            out_elem_kwargs_new = {'out': out_arrays_new[:nout]}

        odl_result_elem_new = elem_fun_new(*in_elems_new,
                                           **out_elem_kwargs_new)
        assert all_almost_equal(npy_result, odl_result_elem_new)

        if nout == 1:
            assert odl_result_elem_new is out_arrays_new[0]
        elif nout > 1:
            for i in range(nout):
                assert odl_result_elem_new[i] is out_arrays_new[i]

    if USE_ARRAY_UFUNCS_INTERFACE:
        # Check `ufunc.at`
        indices = ([0, 0, 1],
                   [0, 1, 2])

        mod_array = in_arrays[0].copy()
        mod_elem = in_elems_new[0].copy()
        if nin == 1:
            npy_result = npy_ufunc.at(mod_array, indices)
            odl_result = npy_ufunc.at(mod_elem, indices)
        elif nin == 2:
            other_array = in_arrays[1][indices]
            other_elem = in_elems_new[1][indices]
            npy_result = npy_ufunc.at(mod_array, indices, other_array)
            odl_result = npy_ufunc.at(mod_elem, indices, other_elem)

        assert all_almost_equal(odl_result, npy_result)

    # Check `ufunc.reduce`
    if nin == 2 and nout == 1 and USE_ARRAY_UFUNCS_INTERFACE:
        in_array = in_arrays[0]
        in_elem = in_elems_new[0]

        # We only test along one axis since some binary ufuncs are not
        # re-orderable, in which case Numpy raises a ValueError
        npy_result = npy_ufunc.reduce(in_array)
        odl_result = npy_ufunc.reduce(in_elem)
        assert all_almost_equal(odl_result, npy_result)
        odl_result_keepdims = npy_ufunc.reduce(in_elem, keepdims=True)
        assert odl_result_keepdims.shape == (1,) + in_elem.shape[1:]
        # In-place using `out` (with ODL vector and array)
        out_elem = odl_result_keepdims.space.element()
        out_array = np.empty(odl_result_keepdims.shape,
                             dtype=odl_result_keepdims.dtype)
        npy_ufunc.reduce(in_elem, out=out_elem, keepdims=True)
        npy_ufunc.reduce(in_elem, out=out_array, keepdims=True)
        assert all_almost_equal(out_elem, odl_result_keepdims)
        assert all_almost_equal(out_array, odl_result_keepdims)
        # Using a specific dtype
        npy_result = npy_ufunc.reduce(in_array, dtype=complex)
        odl_result = npy_ufunc.reduce(in_elem, dtype=complex)
        assert odl_result.dtype == npy_result.dtype
        assert all_almost_equal(odl_result, npy_result)

    # Other ufunc method use the same interface, to we don't perform
    # extra tests for them.


def test_ufunc_corner_cases(odl_tspace_impl):
    """Check if some corner cases are handled correctly."""
    impl = odl_tspace_impl
    space = odl.rn((2, 3), impl=impl)
    x = space.element([[-1, 0, 1],
                       [1, 2, 3]])
    space_const_w = odl.rn((2, 3), weighting=2, impl=impl)
    weights = [[1, 2, 1],
               [3, 2, 1]]
    space_arr_w = odl.rn((2, 3), weighting=weights, impl=impl)

    # --- Ufuncs with nin = 1, nout = 1 --- #

    with pytest.raises(ValueError):
        # Too many arguments
        x.__array_ufunc__(np.sin, '__call__', x, np.ones((2, 3)))

    # Check that `out=(None,)` is the same as not providing `out`
    res = x.__array_ufunc__(np.sin, '__call__', x, out=(None,))
    assert all_almost_equal(res, np.sin(x.asarray()))
    # Check that the result space is the same
    assert res.space == space

    # Check usage of `order` argument
    for order in ('C', 'F'):
        res = x.__array_ufunc__(np.sin, '__call__', x, order=order)
        assert all_almost_equal(res, np.sin(x.asarray()))
        assert res.data.flags[order + '_CONTIGUOUS']

    # Check usage of `dtype` argument
    res = x.__array_ufunc__(np.sin, '__call__', x, dtype='float32')
    assert all_almost_equal(res, np.sin(x.asarray(), dtype='float32'))
    assert res.dtype == 'float32'

    # Check propagation of weightings
    y = space_const_w.one()
    res = y.__array_ufunc__(np.sin, '__call__', y)
    assert res.space.weighting == space_const_w.weighting
    y = space_arr_w.one()
    res = y.__array_ufunc__(np.sin, '__call__', y)
    assert res.space.weighting == space_arr_w.weighting

    # --- Ufuncs with nin = 2, nout = 1 --- #

    with pytest.raises(ValueError):
        # Too few arguments
        x.__array_ufunc__(np.add, '__call__', x)

    with pytest.raises(ValueError):
        # Too many outputs
        out1, out2 = np.empty_like(x), np.empty_like(x)
        x.__array_ufunc__(np.add, '__call__', x, x, out=(out1, out2))

    # Check that npy_array += odl_elem works
    arr = np.ones((2, 3))
    arr += x
    assert all_almost_equal(arr, x.asarray() + 1)
    # For Numpy >= 1.13, this will be equivalent
    arr = np.ones((2, 3))
    res = x.__array_ufunc__(np.add, '__call__', arr, x, out=(arr,))
    assert all_almost_equal(arr, x.asarray() + 1)
    assert res is arr

    # --- `accumulate` --- #

    res = x.__array_ufunc__(np.add, 'accumulate', x)
    assert all_almost_equal(res, np.add.accumulate(x.asarray()))
    assert res.space == space
    arr = np.empty_like(x)
    res = x.__array_ufunc__(np.add, 'accumulate', x, out=(arr,))
    assert all_almost_equal(arr, np.add.accumulate(x.asarray()))
    assert res is arr

    # `accumulate` with other dtype
    res = x.__array_ufunc__(np.add, 'accumulate', x, dtype='float32')
    assert res.dtype == 'float32'

    # Error scenarios
    with pytest.raises(ValueError):
        # Too many `out` arguments
        out1, out2 = np.empty_like(x), np.empty_like(x)
        x.__array_ufunc__(np.add, 'accumulate', x, out=(out1, out2))

    # --- `reduce` --- #

    res = x.__array_ufunc__(np.add, 'reduce', x)
    assert all_almost_equal(res, np.add.reduce(x.asarray()))

    # With `out` argument and `axis`
    out_ax0 = np.empty(3)
    res = x.__array_ufunc__(np.add, 'reduce', x, axis=0, out=(out_ax0,))
    assert all_almost_equal(out_ax0, np.add.reduce(x.asarray(), axis=0))
    assert res is out_ax0
    out_ax1 = odl.rn(2, impl=impl).element()
    res = x.__array_ufunc__(np.add, 'reduce', x, axis=1, out=(out_ax1,))
    assert all_almost_equal(out_ax1, np.add.reduce(x.asarray(), axis=1))
    assert res is out_ax1

    # Addition is reorderable, so we can give multiple axes
    res = x.__array_ufunc__(np.add, 'reduce', x, axis=(0, 1))
    assert res == pytest.approx(np.add.reduce(x.asarray(), axis=(0, 1)))

    # Cannot propagate weightings in a meaningful way, check that there are
    # none in the result
    y = space_const_w.one()
    res = y.__array_ufunc__(np.add, 'reduce', y, axis=0)
    assert not res.space.is_weighted
    y = space_arr_w.one()
    res = y.__array_ufunc__(np.add, 'reduce', y, axis=0)
    assert not res.space.is_weighted

    # Check that `exponent` is propagated
    space_1 = odl.rn((2, 3), exponent=1)
    z = space_1.one()
    res = z.__array_ufunc__(np.add, 'reduce', z, axis=0)
    assert res.space.exponent == 1


def testodl_reduction(tspace, odl_reduction):
    """Test reductions in x.ufunc against direct Numpy reduction."""
    name = odl_reduction
    npy_reduction = getattr(np, name)

    x_arr, x = noise_elements(tspace, 1)
    x_reduction = getattr(x.ufuncs, name)

    # Should be equal theoretically, but summation order, other stuff, ...,
    # hence we use approx

    # Full reduction, produces scalar
    result_npy = npy_reduction(x_arr)
    result = x_reduction()
    assert result == pytest.approx(result_npy)
    result = x_reduction(axis=(0, 1))
    assert result == pytest.approx(result_npy)

    # Reduction along axes, produces element in reduced space
    result_npy = npy_reduction(x_arr, axis=0)
    result = x_reduction(axis=0)
    assert isinstance(result, NumpyTensor)
    assert result.shape == result_npy.shape
    assert result.dtype == x.dtype
    assert np.allclose(result, result_npy)
    # Check reduced space properties
    assert isinstance(result.space, NumpyTensorSpace)
    assert result.space.exponent == x.space.exponent
    assert result.space.weighting == x.space.weighting  # holds true here
    # Evaluate in-place
    out = result.space.element()
    x_reduction(axis=0, out=out)
    assert np.allclose(out, result_npy)

    # Use keepdims parameter
    result_npy = npy_reduction(x_arr, axis=1, keepdims=True)
    result = x_reduction(axis=1, keepdims=True)
    assert result.shape == result_npy.shape
    assert np.allclose(result, result_npy)
    # Evaluate in-place
    out = result.space.element()
    x_reduction(axis=1, keepdims=True, out=out)
    assert np.allclose(out, result_npy)

    # Use dtype parameter
    # These reductions have a `dtype` parameter
    if name in ('cumprod', 'cumsum', 'mean', 'prod', 'std', 'sum',
                'trace', 'var'):
        result_npy = npy_reduction(x_arr, axis=1, dtype='complex64')
        result = x_reduction(axis=1, dtype='complex64')
        assert result.dtype == np.dtype('complex64')
        assert np.allclose(result, result_npy)
        # Evaluate in-place
        out = result.space.element()
        x_reduction(axis=1, dtype='complex64', out=out)
        assert np.allclose(out, result_npy)


def test_ufunc_reduction_docs_notempty(odl_tspace_impl):
    """Check that the generated docstrings are not empty."""
    impl = odl_tspace_impl
    x = odl.rn(3, impl=impl).element()

    for name, _, __, ___ in UFUNCS:
        ufunc = getattr(x.ufuncs, name)
        assert ufunc.__doc__.splitlines()[0] != ''

    for name in ['sum', 'prod', 'min', 'max']:
        reduction = getattr(x.ufuncs, name)
        assert reduction.__doc__.splitlines()[0] != ''


if __name__ == '__main__':
    odl.util.test_file(__file__)
