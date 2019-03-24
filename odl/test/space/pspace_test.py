# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
import pytest

import odl
from odl.space import ProductSpace
from odl.util.testutils import (
    all_equal, all_almost_equal, noise_elements, noise_element, simple_fixture)


# --- Helpers --- #


def _inner(x1, x2, w):
    if w is None:
        w = 1.0
    inners = np.array([np.vdot(x2_i, x1_i) for x1_i, x2_i in zip(x1, x2)])
    return np.sum(w * inners)


def _norm(x, p, w):
    if w is None:
        w = 1.0
    norms = np.array([np.linalg.norm(xi.ravel()) for xi in x])
    if p in {float('inf'), 0.0, -float('inf')}:
        return np.linalg.norm(norms.ravel(), p)
    else:
        w = np.asarray(w, dtype=float)
        return np.linalg.norm((w ** (1 / p) * norms).ravel(), p)


def _dist(x1, x2, p, w):
    return _norm([x1_i - x2_i for x1_i, x2_i in zip(x1, x2)], p, w)


# --- pytest Fixtures --- #


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.0, 1.5])
weighting = simple_fixture('weighting', [None, 2.0, [1.5, 2.5]])

space_params = ['product_space', 'power_space']
space_ids = [' space={} '.format(p) for p in space_params]

elem_params = [
    'space', 'real_space', 'numpy_array', 'array', 'scalar', '1d_array'
]
elem_ids = [' element={} '.format(p) for p in elem_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def space(request):
    name = request.param.strip()

    if name == 'product_space':
        space = odl.ProductSpace(odl.uniform_discr(0, 1, 3, dtype=complex),
                                 odl.cn(2))
    elif name == 'power_space':
        space = odl.ProductSpace(odl.uniform_discr(0, 1, 3, dtype=complex), 2)
    else:
        raise ValueError('undefined space')

    return space


# --- Tests --- #


def test_init_pspace():
    """Test initialization patterns and options for ``ProductSpace``."""
    r2 = odl.rn(2)
    r3 = odl.rn(3)

    ProductSpace(r2, r3)
    ProductSpace(r2, r3, r3, r2)
    ProductSpace(r2, r3, exponent=1.0)
    ProductSpace(r2, r3, field=odl.RealNumbers())
    ProductSpace(r2, r3, weighting=0.5)
    ProductSpace(r2, r3, weighting=[0.5, 2])
    ProductSpace(r2, 4)

    r2 * r3
    r2 ** 3

    # Make sure `repr` at works at least in the very basic case
    assert repr(ProductSpace(r2, r3)) != ''


def test_empty_pspace():
    """Test that empty product spaces have sensible behavior."""
    with pytest.raises(ValueError):
        # Requires explicit `field`
        odl.ProductSpace()

    field = odl.RealNumbers()
    spc = odl.ProductSpace(field=field)
    assert spc.field == field
    assert spc.size == 0
    assert spc.is_real
    assert spc.is_complex

    with pytest.raises(IndexError):
        spc[0]


def test_pspace_basic_properties():
    """Verify basic properties of product spaces."""
    r2 = odl.rn(2)
    r3 = odl.rn(3)

    # Non-power space
    pspace = odl.ProductSpace(r2, r3)
    assert len(pspace) == 2
    assert pspace.shape == (2,)
    assert pspace.size == 2
    assert pspace.spaces[0] == r2
    assert pspace.spaces[1] == r3
    assert not pspace.is_power_space
    assert not pspace.is_weighted
    assert pspace.is_real
    assert not pspace.is_complex

    r2_x = r2.element([1, 2])
    r3_x = r3.element([3, 4, 5])
    x = pspace.element([r2_x, r3_x])
    y = pspace.element([[1, 2], [3, 4, 5]])
    assert all_equal(x, y)
    assert all_equal(x, [r2_x, r3_x])

    # Power space
    pspace = odl.ProductSpace(r3, 2)
    assert len(pspace) == 2
    assert pspace.shape == (2,)
    assert pspace.size == 2
    assert pspace.spaces[0] == pspace.spaces[1] == r3
    assert pspace.is_power_space
    assert not pspace.is_weighted
    assert pspace.is_real
    assert not pspace.is_complex

    x1 = r3.element([0, 1, 2])
    x2 = r3.element([3, 4, 5])
    x = pspace.element([x1, x2])
    y = pspace.element([[0, 1, 2], [3, 4, 5]])
    assert all_equal(x, y)
    assert all_equal(x, [x1, x2])


def test_pspace_equality(exponent):
    """Verify equality checking of product spaces."""
    r2 = odl.rn(2)
    pspace_1 = odl.ProductSpace(r2, 3, exponent=exponent)
    pspace_1_same = odl.ProductSpace(r2, 3, exponent=exponent)
    pspace_2 = odl.ProductSpace(r2, 4, exponent=exponent)
    pspace_3 = odl.ProductSpace(
        r2, 3, exponent=1.0 if exponent != 1.0 else 2.0
    )

    assert pspace_1 == pspace_1
    assert pspace_1 == pspace_1_same
    assert pspace_1 != pspace_2
    assert pspace_1 != pspace_3
    assert hash(pspace_1) == hash(pspace_1_same)
    assert hash(pspace_1) != hash(pspace_2)
    assert hash(pspace_1) != hash(pspace_3)


# TODO(kohr-h): higher-order spaces
def test_pspace_element():
    """Test element creation in product spaces."""
    r2 = odl.rn(2)
    pspace = odl.ProductSpace(r2, r2)
    x = pspace.element([[1, 2], [3, 4]])
    assert x in pspace

    # Wrong length
    with pytest.raises(ValueError):
        pspace.element([[1, 2]])

    with pytest.raises(ValueError):
        pspace.element([[1, 2], [3, 4], [5, 6]])

    # Wrong length of subspace element
    with pytest.raises(ValueError):
        pspace.element([[1, 2, 3], [4, 5]])

    with pytest.raises(ValueError):
        pspace.element([[1, 2], [3, 4, 5]])


def test_pspace_lincomb():
    """Test linear combination in product spaces."""
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, H)

    v1 = H.element([1, 2])
    v2 = H.element([5, 3])
    u1 = H.element([-1, 7])
    u2 = H.element([2, 1])

    v = HxH.element([v1, v2])
    u = HxH.element([u1, u2])
    z = HxH.element()

    a = 3.12
    b = 1.23

    expected = [a * v1 + b * u1, a * v2 + b * u2]
    HxH.lincomb(a, v, b, u, out=z)

    assert all_almost_equal(z, expected)


def test_pspace_multiply():
    """Test multiplication in product spaces."""
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, H)

    v1 = H.element([1, 2])
    v2 = H.element([5, 3])
    u1 = H.element([-1, 7])
    u2 = H.element([2, 1])

    v = HxH.element([v1, v2])
    u = HxH.element([u1, u2])
    z = HxH.element()

    expected = [v1 * u1, v2 * u2]
    HxH.multiply(v, u, out=z)

    assert all_almost_equal(z, expected)


def test_pspace_dist(exponent, weighting):
    """Test product space distance function implementation."""
    r2 = odl.rn(2)
    r3 = odl.rn(3)
    pspace = odl.ProductSpace(r2, r3, exponent=exponent, weighting=weighting)

    (x1_arr, x2_arr), (x1, x2) = noise_elements(r2, 2)
    (y1_arr, y2_arr), (y1, y2) = noise_elements(r3, 2)
    z1 = pspace.element([x1, y1])
    z2 = pspace.element([x2, y2])

    true_dist = _dist([x1_arr, y1_arr], [x2_arr, y2_arr], exponent, weighting)
    assert pspace.dist(z1, z2) == pytest.approx(true_dist)


def test_pspace_norm(exponent, weighting):
    """Test product space norm implementation."""
    r2 = odl.rn(2)
    r3 = odl.rn(3)
    pspace = odl.ProductSpace(r2, r3, exponent=exponent, weighting=weighting)

    x_arr, x = noise_elements(r2)
    y_arr, y = noise_elements(r3)
    z = pspace.element([x, y])

    true_norm = _norm([x_arr, y_arr], exponent, weighting)
    assert pspace.norm(z) == pytest.approx(true_norm)


def test_pspace_inner(weighting):
    """Test product space inner product implementation."""
    r2 = odl.rn(2)
    r3 = odl.rn(3)
    pspace = odl.ProductSpace(r2, r3, weighting=weighting)

    (x1_arr, x2_arr), (x1, x2) = noise_elements(r2, 2)
    (y1_arr, y2_arr), (y1, y2) = noise_elements(r3, 2)
    z1 = pspace.element([x1, y1])
    z2 = pspace.element([x2, y2])

    true_inner = _inner([x1_arr, y1_arr], [x2_arr, y2_arr], weighting)
    assert pspace.inner(z1, z2) == pytest.approx(true_inner)


def _test_shape(space, expected_shape):
    """Helper to validate shape of space."""
    space_el = space.element()

    assert space.shape == expected_shape
    assert space_el.shape == expected_shape
    assert space.size == np.prod(expected_shape)
    assert space_el.size == np.prod(expected_shape)
    assert len(space) == expected_shape[0]
    assert len(space_el) == expected_shape[0]


def test_power_shape():
    """Check if shape and size are correct for higher-order power spaces."""
    r2 = odl.rn(2)
    r3 = odl.rn(3)

    empty = odl.ProductSpace(field=odl.RealNumbers())
    empty2 = odl.ProductSpace(r2, 0)
    assert empty.shape == empty2.shape == ()
    assert empty.size == empty2.size == 0

    r2_3 = odl.ProductSpace(r2, 3)
    _test_shape(r2_3, (3,))

    r2xr3 = odl.ProductSpace(r2, r3)
    _test_shape(r2xr3, (2,))

    r2xr3_4 = odl.ProductSpace(r2xr3, 4)
    _test_shape(r2xr3_4, (4, 2))

    r2xr3_5_4 = odl.ProductSpace(r2xr3_4, 5)
    _test_shape(r2xr3_5_4, (5, 4, 2))


def test_power_lincomb():
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, 2)

    v1 = H.element([1, 2])
    v2 = H.element([5, 3])
    u1 = H.element([-1, 7])
    u2 = H.element([2, 1])

    v = HxH.element([v1, v2])
    u = HxH.element([u1, u2])
    z = HxH.element()

    a = 3.12
    b = 1.23

    expected = [a * v1 + b * u1, a * v2 + b * u2]
    HxH.lincomb(a, v, b, u, out=z)

    assert all_almost_equal(z, expected)


def test_power_in_place_modify():
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, 2)

    v1 = H.element([1, 2])
    v2 = H.element([5, 3])
    u1 = H.element([-1, 7])
    u2 = H.element([2, 1])
    z1 = H.element()
    z2 = H.element()

    v = HxH.element([v1, v2])
    u = HxH.element([u1, u2])
    z = HxH.element([z1, z2])  # z is simply a wrapper for z1 and z2

    a = 3.12
    b = 1.23

    HxH.lincomb(a, v, b, u, out=z)

    # Assert that z1 and z2 has been modified as well
    assert all_almost_equal(z, [z1, z2])


def test_getitem_single():
    r1 = odl.rn(1)
    r2 = odl.rn(2)
    H = odl.ProductSpace(r1, r2)

    assert H[-2] is r1
    assert H[-1] is r2
    assert H[0] is r1
    assert H[1] is r2
    with pytest.raises(IndexError):
        H[-3]
        H[2]

    assert H[(1,)] == r2
    with pytest.raises(IndexError):
        H[0, 1]


def test_getitem_slice():
    r1 = odl.rn(1)
    r2 = odl.rn(2)
    r3 = odl.rn(3)
    H = odl.ProductSpace(r1, r2, r3)

    assert H[:2] == odl.ProductSpace(r1, r2)
    assert H[:2][0] is r1
    assert H[:2][1] is r2

    assert H[3:] == odl.ProductSpace(field=r1.field)


def test_getitem_fancy():
    r1 = odl.rn(1)
    r2 = odl.rn(2)
    r3 = odl.rn(3)
    H = odl.ProductSpace(r1, r2, r3)

    assert H[[0, 2]] == odl.ProductSpace(r1, r3)
    assert H[[0, 2]][0] is r1
    assert H[[0, 2]][1] is r3


# TODO(kohr-h): ufunc tests


if __name__ == '__main__':
    odl.util.test_file(__file__)
