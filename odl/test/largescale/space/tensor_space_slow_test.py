# Copyright 2014-2017 The ODL contributors
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
from odl.util.testutils import all_almost_equal, almost_equal, noise_elements

pytestmark = odl.util.skip_if_no_largescale


# Pytest fixtures
spc_params = ['rn', '1d', '3d']
spc_ids = [' type={} ' ''.format(p) for p in spc_params]


@pytest.fixture(scope="module", ids=spc_ids, params=spc_params)
def tspace(tspace_impl, request):
    spc = request.param

    if spc == 'rn':
        return odl.rn(10 ** 5, impl=tspace_impl)
    elif spc == '1d':
        return odl.uniform_discr(0, 1, 10 ** 5, impl=tspace_impl)
    elif spc == '3d':
        return odl.uniform_discr([0, 0, 0], [1, 1, 1],
                                 [100, 100, 100], impl=tspace_impl)


def test_element(tspace):
    x = tspace.element()
    assert x in tspace

    y = tspace.element(inp=[0] * tspace.size)
    assert y in tspace

    # Rewrap
    y2 = tspace.element(y)
    assert y2 is y

    w = tspace.element(inp=np.zeros(tspace.size, tspace.dtype))
    assert w in tspace


def test_zero(tspace):
    assert np.allclose(tspace.zero(), 0)


def test_one(tspace):
    assert np.allclose(tspace.one(), 1)


def test_ndarray_init(tspace):
    x0 = np.arange(tspace.size)
    x = tspace.element(x0)

    assert all_almost_equal(x0, x)


def test_getitem(tspace):
    indices = np.random.randint(0, tspace.size - 1, 5)

    x0 = np.arange(tspace.size)
    x = tspace.element(x0)

    for index in indices:
        assert x[index] == index


def test_setitem(tspace):
    indices = np.random.randint(0, tspace.size - 1, 5)

    x0 = np.arange(tspace.size)
    x = tspace.element(x0)

    for index in indices:
        x[index] = -index
        assert x[index] == -index


def tspace_weighting(tspace):
    """ Get the weighting of a tensor space """

    # TODO: use actual weighting

    if isinstance(tspace, odl.DiscreteLp):
        return tspace.domain.volume / tspace.size
    else:
        return 1.0


def test_inner(tspace):
    weighting = tspace_weighting(tspace)

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    correct_inner = np.vdot(yarr, xarr) * weighting

    assert almost_equal(tspace.inner(x, y), correct_inner, places=2)
    assert almost_equal(x.inner(y), correct_inner, places=2)


def test_norm(tspace):
    weighting = np.sqrt(tspace_weighting(tspace))

    xarr, x = noise_elements(tspace)

    correct_norm = np.linalg.norm(xarr) * weighting

    assert almost_equal(tspace.norm(x), correct_norm, places=2)
    assert almost_equal(x.norm(), correct_norm, places=2)


def test_dist(tspace):
    weighting = np.sqrt(tspace_weighting(tspace))

    [xarr, yarr], [x, y] = noise_elements(tspace, 2)

    correct_dist = np.linalg.norm(xarr - yarr) * weighting

    assert almost_equal(tspace.dist(x, y), correct_dist, places=2)
    assert almost_equal(x.dist(y), correct_dist, places=2)


def _test_lincomb(tspace, a, b):
    # Validates lincomb against the result on host with randomized
    # data and given a,b

    # Unaliased arguments
    [x_arr, y_arr, z_arr], [x, y, z] = noise_elements(tspace, 3)

    z_arr[:] = a * x_arr + b * y_arr
    z.lincomb(a, x, b, y)

    order = getattr(z, 'order', None)
    assert all_almost_equal(z.asarray().ravel(order), z_arr, places=2)


def test_lincomb(tspace):
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(tspace, a, b)


def _test_member_lincomb(spc, a):
    # Validates vector member lincomb against the result on host

    # Generate vectors
    [x_host, y_host], [x_device, y_device] = noise_elements(spc, 2)

    # Host side calculation
    y_host[:] = a * x_host

    # Device side calculation
    y_device.lincomb(a, x_device)

    # CUDA only uses floats, so require 5 places
    assert all_almost_equal(y_device, y_host, places=2)


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
    pytest.main([str(__file__.replace('\\', '/')), '-v', '--largescale'])
