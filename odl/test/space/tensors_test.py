# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Unit tests for Numpy-based tensors."""

from __future__ import division

import numpy as np
import pytest

import odl
from odl.space.npy_tensors import NumpyTensorSpace
from odl.util.testutils import (
    all_almost_equal, all_equal, noise_array, noise_elements, simple_fixture)

# --- Helpers --- #


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
        return np.linalg.norm(x.ravel(), p)
    else:
        return np.linalg.norm((w ** (1 / p) * x).ravel(), p)


def _dist(x1, x2, p, w):
    return _norm(x1 - x2, p, w)


# --- pytest Fixtures --- #


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])

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


# --- Tests --- #


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
    assert elem.shape == tspace.shape
    assert elem.dtype == tspace.dtype
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
        res_space = space[slc]
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


@pytest.mark.xfail(reason='need space indexing to test this')
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


def test_lincomb_exceptions(tspace):
    """Test whether lincomb raises correctly for bad output element."""
    other_space = odl.rn((4, 3), impl=tspace.impl)

    wrong_out = other_space.zero()
    x, y = tspace.zero(), tspace.zero()

    with pytest.raises(TypeError):
        # Only `out` must be an element of the space, thus raising TypeError
        tspace.lincomb(1, x, 1, y, wrong_out)


def test_multiply(tspace):
    """Test multiply against direct array multiplication."""
    # space method
    [x_arr, y_arr, out_arr], [x, y, out] = noise_elements(tspace, 3)
    out_arr = x_arr * y_arr

    tspace.multiply(x, y, out)
    assert all_almost_equal([x_arr, y_arr, out_arr], [x, y, out])


def test_multiply_exceptions(tspace):
    """Test if multiply raises correctly for bad output element."""
    other_space = odl.rn((4, 3))

    wrong_out = other_space.zero()
    x, y = tspace.zero(), tspace.zero()

    with pytest.raises(TypeError):
        tspace.multiply(x, y, wrong_out)


def test_inner(tspace):
    """Test the inner method against numpy.vdot."""
    rel = 1e-2 if tspace.dtype == 'float16' else 1e-5
    (xarr, yarr), (x, y) = noise_elements(tspace, 2)
    correct_inner = _inner(xarr, yarr, tspace.weighting)
    assert tspace.inner(x, y) == pytest.approx(correct_inner, rel=rel)


def test_norm(tspace):
    """Test the norm method against numpy.linalg.norm."""
    rel = 1e-2 if tspace.dtype == 'float16' else 1e-5
    xarr, x = noise_elements(tspace)
    correct_norm = _norm(xarr, tspace.exponent, tspace.weighting)
    assert tspace.norm(x) == pytest.approx(correct_norm, rel=rel)


def test_pnorm(exponent):
    """Test the norm method with p!=2 against numpy.linalg.norm."""
    for tspace in (odl.rn((3, 4), exponent=exponent),
                   odl.cn((3, 4), exponent=exponent)):
        xarr, x = noise_elements(tspace)
        correct_norm = _norm(xarr, exponent, 1.0)
        assert tspace.norm(x) == pytest.approx(correct_norm)


def test_dist(tspace):
    """Test the dist method against numpy.linalg.norm of the difference."""
    rel = 1e-2 if tspace.dtype == 'float16' else 1e-5
    (xarr, yarr), (x, y) = noise_elements(tspace, n=2)
    correct_dist = _dist(xarr, yarr, tspace.exponent, tspace.weighting)
    assert tspace.dist(x, y) == pytest.approx(correct_dist, rel=rel)


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
