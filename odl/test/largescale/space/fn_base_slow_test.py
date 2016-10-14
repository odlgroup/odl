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

"""Test to make sure the FnBase spaces work with larger sizes."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl.util.testutils import all_almost_equal, almost_equal, noise_elements

pytestmark = odl.util.skip_if_no_largescale


# Pytest fixtures
spc_params = ['rn', '1d', '3d']
spc_ids = [' type={} ' ''.format(p) for p in spc_params]


@pytest.fixture(scope="module", ids=spc_ids, params=spc_params)
def fn(fn_impl, request):
    spc = request.param

    if spc == 'rn':
        return odl.rn(10 ** 5, impl=fn_impl)
    elif spc == '1d':
        return odl.uniform_discr(0, 1, 10 ** 5, impl=fn_impl)
    elif spc == '3d':
        return odl.uniform_discr([0, 0, 0], [1, 1, 1],
                                 [100, 100, 100], impl=fn_impl)


def test_element(fn):
    x = fn.element()
    assert x in fn

    y = fn.element(inp=[0] * fn.size)
    assert y in fn

    # Rewrap
    y2 = fn.element(y)
    assert y2 is y

    w = fn.element(inp=np.zeros(fn.size, fn.dtype))
    assert w in fn


def test_zero(fn):
    assert np.allclose(fn.zero(), 0)


def test_one(fn):
    assert np.allclose(fn.one(), 1)


def test_ndarray_init(fn):
    x0 = np.arange(fn.size)
    x = fn.element(x0)

    assert all_almost_equal(x0, x)


def test_getitem(fn):
    indices = np.random.randint(0, fn.size - 1, 5)

    x0 = np.arange(fn.size)
    x = fn.element(x0)

    for index in indices:
        assert x[index] == index


def test_setitem(fn):
    indices = np.random.randint(0, fn.size - 1, 5)

    x0 = np.arange(fn.size)
    x = fn.element(x0)

    for index in indices:
        x[index] = -index
        assert x[index] == -index


def fn_weighting(fn):
    """ Get the weighting of a fn space """

    # TODO: use actual weighting

    if isinstance(fn, odl.DiscreteLp):
        return fn.domain.volume / fn.size
    else:
        return 1.0


def test_inner(fn):
    weighting = fn_weighting(fn)

    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    correct_inner = np.vdot(yarr, xarr) * weighting

    assert almost_equal(fn.inner(x, y), correct_inner, places=2)
    assert almost_equal(x.inner(y), correct_inner, places=2)


def test_norm(fn):
    weighting = np.sqrt(fn_weighting(fn))

    xarr, x = noise_elements(fn)

    correct_norm = np.linalg.norm(xarr) * weighting

    assert almost_equal(fn.norm(x), correct_norm, places=2)
    assert almost_equal(x.norm(), correct_norm, places=2)


def test_dist(fn):
    weighting = np.sqrt(fn_weighting(fn))

    [xarr, yarr], [x, y] = noise_elements(fn, 2)

    correct_dist = np.linalg.norm(xarr - yarr) * weighting

    assert almost_equal(fn.dist(x, y), correct_dist, places=2)
    assert almost_equal(x.dist(y), correct_dist, places=2)


def _test_lincomb(fn, a, b):
    # Validates lincomb against the result on host with randomized
    # data and given a,b

    # Unaliased arguments
    [x_arr, y_arr, z_arr], [x, y, z] = noise_elements(fn, 3)

    z_arr[:] = a * x_arr + b * y_arr
    z.lincomb(a, x, b, y)

    order = getattr(z, 'order', None)
    assert all_almost_equal(z.asarray().ravel(order), z_arr, places=2)


def test_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41]
    for a in scalar_values:
        for b in scalar_values:
            _test_lincomb(fn, a, b)


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


def test_member_lincomb(fn):
    scalar_values = [0, 1, -1, 3.41, 10.0, 1.0001]
    for a in scalar_values:
        _test_member_lincomb(fn, a)


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


def test_operators(fn):
    # Test of all operator overloads against the corresponding
    # Numpy implementation

    # Unary operators
    _test_unary_operator(fn, lambda x: +x)
    _test_unary_operator(fn, lambda x: -x)

    # Scalar multiplication
    for scalar in [-31.2, -1, 0, 1, 2.13]:
        def imul(x):
            x *= scalar
        _test_unary_operator(fn, imul)
        _test_unary_operator(fn, lambda x: x * scalar)

    # Scalar division
    for scalar in [-31.2, -1, 1, 2.13]:
        def idiv(x):
            x /= scalar
        _test_unary_operator(fn, idiv)
        _test_unary_operator(fn, lambda x: x / scalar)

    # Incremental operations
    def iadd(x, y):
        x += y

    def isub(x, y):
        x -= y

    def imul(x, y):
        x *= y

    def idiv(x, y):
        x /= y

    _test_binary_operator(fn, iadd)
    _test_binary_operator(fn, isub)
    _test_binary_operator(fn, imul)
    _test_binary_operator(fn, idiv)

    # Incremental operators with aliased inputs
    def iadd_aliased(x):
        x += x

    def isub_aliased(x):
        x -= x

    def imul_aliased(x):
        x *= x

    def idiv_aliased(x):
        x /= x

    _test_unary_operator(fn, iadd_aliased)
    _test_unary_operator(fn, isub_aliased)
    _test_unary_operator(fn, imul_aliased)
    _test_unary_operator(fn, idiv_aliased)

    # Binary operators
    _test_binary_operator(fn, lambda x, y: x + y)
    _test_binary_operator(fn, lambda x, y: x - y)
    _test_binary_operator(fn, lambda x, y: x * y)
    _test_binary_operator(fn, lambda x, y: x / y)

    # Binary with aliased inputs
    _test_unary_operator(fn, lambda x: x + x)
    _test_unary_operator(fn, lambda x: x - x)
    _test_unary_operator(fn, lambda x: x * x)
    _test_unary_operator(fn, lambda x: x / x)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v', '--largescale'])
