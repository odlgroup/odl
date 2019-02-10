# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test to make sure the `TensorSpace` spaces work with larger sizes."""

from __future__ import division

import numpy as np
import pytest

import odl
from odl.util.testutils import (
    all_almost_equal, dtype_tol, noise_elements, skip_if_no_largescale)


# --- pytest fixtures --- #


pytestmark = skip_if_no_largescale


spc_params = ['rn', '1d', '3d']
spc_ids = [' type={} '.format(p) for p in spc_params]


@pytest.fixture(scope="module", ids=spc_ids, params=spc_params)
def tspace(odl_tspace_impl, request):
    spc = request.param
    impl = odl_tspace_impl

    if spc == 'rn':
        return odl.rn(10 ** 5, impl=impl)
    elif spc == '1d':
        return odl.uniform_discr(0, 1, 10 ** 5, impl=impl)
    elif spc == '3d':
        return odl.uniform_discr([0, 0, 0], [1, 1, 1],
                                 [100, 100, 100], impl=impl)


def test_element(tspace):
    x = tspace.element()
    assert x in tspace

    # From array-like
    y = tspace.element(np.zeros(tspace.shape, dtype=tspace.dtype).tolist())
    assert y in tspace

    # Rewrap
    y2 = tspace.element(y)
    assert y2 is y

    w = tspace.element(np.zeros(tspace.shape, dtype=tspace.dtype))
    assert w in tspace


def test_zero(tspace):
    assert np.allclose(tspace.zero(), 0)


def test_one(tspace):
    assert np.allclose(tspace.one(), 1)


def test_ndarray_init(tspace):
    x0 = np.arange(tspace.size).reshape(tspace.shape)
    x = tspace.element(x0)

    assert all_almost_equal(x0, x)


def test_getitem(tspace):
    indices = np.random.randint(0, tspace.size - 1, 5)
    indices = np.unravel_index(indices, tspace.shape)

    x0 = np.arange(tspace.size).reshape(tspace.shape)
    x = tspace.element(x0)

    for index in zip(*indices):
        assert x[index] == np.ravel_multi_index(index, tspace.shape)


def test_setitem(tspace):
    indices = np.random.randint(0, tspace.size - 1, 5)
    indices = np.unravel_index(indices, tspace.shape)

    x = tspace.zero()

    for index in zip(*indices):
        flat_index = np.ravel_multi_index(index, tspace.shape)
        x[index] = -flat_index
        assert x[index] == -flat_index


def test_inner(tspace):
    weighting_const = tspace.weighting.const

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    correct_inner = np.vdot(yarr, xarr) * weighting_const

    assert (
        tspace.inner(x, y)
        == pytest.approx(correct_inner, rel=dtype_tol(tspace.dtype))
    )
    assert (
        x.inner(y)
        == pytest.approx(correct_inner, rel=dtype_tol(tspace.dtype))
    )


def test_norm(tspace):
    weighting_const = tspace.weighting.const

    xarr, x = noise_elements(tspace)

    correct_norm = np.linalg.norm(xarr) * np.sqrt(weighting_const)

    assert (
        tspace.norm(x)
        == pytest.approx(correct_norm, rel=dtype_tol(tspace.dtype))
    )
    assert (
        x.norm()
        == pytest.approx(correct_norm, rel=dtype_tol(tspace.dtype))
    )


def test_dist(tspace):
    weighting_const = tspace.weighting.const

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    correct_dist = np.linalg.norm(xarr - yarr) * np.sqrt(weighting_const)

    assert (
        tspace.dist(x, y)
        == pytest.approx(correct_dist, rel=dtype_tol(tspace.dtype))
    )
    assert (
        x.dist(y)
        == pytest.approx(correct_dist, rel=dtype_tol(tspace.dtype))
    )


def _test_lincomb(space, a, b, discontig):
    """Validate lincomb against direct result using arrays."""
    # Set slice for discontiguous arrays and get result space of slicing
    if discontig:
        slc = tuple([slice(None)] * (space.ndim - 1) + [slice(None, None, 2)])
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
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b, discontig=False)
            _test_lincomb(tspace, a, b, discontig=True)


def _test_member_lincomb(spc, a):
    # Validates vector member lincomb against the result on host

    # Generate vectors
    [x_host, y_host], [x_device, y_device] = noise_elements(spc, 2)

    # Host side calculation
    y_host[:] = a * x_host

    # Device side calculation
    y_device.lincomb(a, x_device)

    # CUDA only uses floats, so require 2 digits
    assert all_almost_equal(y_device, y_host, ndigits=2)


def test_member_lincomb(tspace):
    scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
    for a in scalar_values:
        _test_member_lincomb(tspace, a)


def _test_unary_operator(spc, function):
    # Verify that the statement y=function(x) gives equivalent
    # results to Numpy.
    x_arr, x = noise_elements(spc)

    y_arr = function(x_arr)
    y = function(x)

    assert all_almost_equal([x, y],
                            [x_arr, y_arr])


def _test_binary_operator(spc, function):
    # Verify that the statement z=function(x,y) gives equivalent
    # results to Numpy.
    [x_arr, y_arr], [x, y] = noise_elements(spc, 2)

    z_arr = function(x_arr, y_arr)
    z = function(x, y)

    assert all_almost_equal([x, y, z],
                            [x_arr, y_arr, z_arr])


def test_operators(tspace):
    # Test of all operator overloads against the corresponding
    # Numpy implementation

    # Unary operators
    _test_unary_operator(tspace, lambda x: +x)
    _test_unary_operator(tspace, lambda x: -x)

    # Scalar multiplication
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def imul(x):
            x *= scalar
        _test_unary_operator(tspace, imul)
        _test_unary_operator(tspace, lambda x: x * scalar)

    # Scalar division
    for scalar in [-31.2, -1, 1, 2.13]:
        def idiv(x):
            x /= scalar
        _test_unary_operator(tspace, idiv)
        _test_unary_operator(tspace, lambda x: x / scalar)

    # Incremental operations
    def iadd(x, y):
        x += y

    def isub(x, y):
        x -= y

    def imul(x, y):
        x *= y

    def idiv(x, y):
        x /= y

    _test_binary_operator(tspace, iadd)
    _test_binary_operator(tspace, isub)
    _test_binary_operator(tspace, imul)
    _test_binary_operator(tspace, idiv)

    # Incremental operators with aliased inputs
    def iadd_aliased(x):
        x += x

    def isub_aliased(x):
        x -= x

    def imul_aliased(x):
        x *= x

    def idiv_aliased(x):
        x /= x

    _test_unary_operator(tspace, iadd_aliased)
    _test_unary_operator(tspace, isub_aliased)
    _test_unary_operator(tspace, imul_aliased)
    _test_unary_operator(tspace, idiv_aliased)

    # Binary operators
    _test_binary_operator(tspace, lambda x, y: x + y)
    _test_binary_operator(tspace, lambda x, y: x - y)
    _test_binary_operator(tspace, lambda x, y: x * y)
    _test_binary_operator(tspace, lambda x, y: x / y)

    # Binary with aliased inputs
    _test_unary_operator(tspace, lambda x: x + x)
    _test_unary_operator(tspace, lambda x: x - x)
    _test_unary_operator(tspace, lambda x: x * x)
    _test_unary_operator(tspace, lambda x: x / x)


if __name__ == '__main__':
    odl.util.test_file(__file__)
