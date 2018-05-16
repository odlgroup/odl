# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import numpy as np
import pytest
import operator

import odl
from odl.util.testutils import (all_equal, all_almost_equal, almost_equal,
                                noise_elements, noise_element, simple_fixture)


exponent = simple_fixture('exponent', [2.0, 1.0, float('inf'), 0.5, 1.5])

space_params = ['product_space', 'power_space']
space_ids = [' space={} '.format(p) for p in space_params]

elem_params = ['space', 'real_space', 'numpy_array', 'array', 'scalar',
               '1d_array']
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


@pytest.fixture(scope="module", ids=elem_ids, params=elem_params)
def newpart(request, space):
    element_form = request.param.strip()

    if element_form == 'space':
        tmp = noise_element(space)
        newreal = space.element(tmp.real)
    elif element_form == 'real_space':
        newreal = noise_element(space).real
    elif element_form == 'numpy_array':
        tmp = noise_element(space)
        newreal = [tmp[0].real.asarray(), tmp[1].real.asarray()]
    elif element_form == 'array':
        if space.is_power_space:
            newreal = [[0, 1, 2], [3, 4, 5]]
        else:
            newreal = [[0, 1, 2], [3, 4]]
    elif element_form == 'scalar':
        newreal = np.random.randn()
    elif element_form == '1d_array':
        if not space.is_power_space:
            pytest.skip('arrays matching only one dimension can only be used '
                        'for power spaces')
        newreal = [0, 1, 2]
    else:
        raise ValueError('undefined form of element')

    return newreal


def test_emptyproduct():
    with pytest.raises(ValueError):
        odl.ProductSpace()

    reals = odl.RealNumbers()
    spc = odl.ProductSpace(field=reals)
    assert spc.field == reals
    assert spc.size == 0
    assert spc.is_real
    assert spc.is_complex

    with pytest.raises(IndexError):
        spc[0]


def test_RxR():
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, H)

    # Check the basic properties
    assert len(HxH) == 2
    assert HxH.shape == (2, 2)
    assert HxH.size == 4
    assert HxH.dtype == H.dtype
    assert HxH.spaces[0] is H
    assert HxH.spaces[1] is H
    assert HxH.is_power_space
    assert not HxH.is_weighted
    assert HxH.is_real
    assert not HxH.is_complex

    v1 = H.element([1, 2])
    v2 = H.element([3, 4])
    v = HxH.element([v1, v2])
    u = HxH.element([[1, 2], [3, 4]])

    assert all_equal([v1, v2], v)
    assert all_equal([v1, v2], u)


def test_equals_space(exponent):
    r2 = odl.rn(2)
    r2x3_1 = odl.ProductSpace(r2, 3, exponent=exponent)
    r2x3_2 = odl.ProductSpace(r2, 3, exponent=exponent)
    r2x4 = odl.ProductSpace(r2, 4, exponent=exponent)

    assert r2x3_1 is r2x3_1
    assert r2x3_1 is not r2x3_2
    assert r2x3_1 is not r2x4
    assert r2x3_1 == r2x3_1
    assert r2x3_1 == r2x3_2
    assert r2x3_1 != r2x4
    assert hash(r2x3_1) == hash(r2x3_2)
    assert hash(r2x3_1) != hash(r2x4)


def test_equals_vec(exponent):
    r2 = odl.rn(2)
    r2x3 = odl.ProductSpace(r2, 3, exponent=exponent)
    r2x4 = odl.ProductSpace(r2, 4, exponent=exponent)

    x1 = r2x3.zero()
    x2 = r2x3.zero()
    y = r2x3.one()
    z = r2x4.zero()

    assert x1 is x1
    assert x1 is not x2
    assert x1 is not y
    assert x1 == x1
    assert x1 == x2
    assert x1 != y
    assert x1 != z


def test_is_power_space():
    r2 = odl.rn(2)
    r2x3 = odl.ProductSpace(r2, 3)
    assert len(r2x3) == 3
    assert r2x3.is_power_space
    assert r2x3.spaces[0] is r2
    assert r2x3.spaces[1] is r2
    assert r2x3.spaces[2] is r2

    r2r2r2 = odl.ProductSpace(r2, r2, r2)
    assert r2x3 == r2r2r2


def test_mixed_space():
    """Verify that a mixed productspace is handled properly."""
    r2_1 = odl.rn(2, dtype='float64')
    r2_2 = odl.rn(2, dtype='float32')
    pspace = odl.ProductSpace(r2_1, r2_2)

    assert not pspace.is_power_space
    assert pspace.spaces[0] is r2_1
    assert pspace.spaces[1] is r2_2
    assert pspace.is_real
    assert not pspace.is_complex

    # dtype not well defined for this space
    with pytest.raises(AttributeError):
        pspace.dtype


def test_element():
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, H)

    HxH.element([[1, 2], [3, 4]])

    # wrong length
    with pytest.raises(ValueError):
        HxH.element([[1, 2]])

    with pytest.raises(ValueError):
        HxH.element([[1, 2], [3, 4], [5, 6]])

    # wrong length of subspace element
    with pytest.raises(ValueError):
        HxH.element([[1, 2, 3], [4, 5]])

    with pytest.raises(ValueError):
        HxH.element([[1, 2], [3, 4, 5]])


def test_lincomb():
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


def test_multiply():
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


def test_metric():
    H = odl.rn(2)
    v11 = H.element([1, 2])
    v12 = H.element([5, 3])

    v21 = H.element([1, 2])
    v22 = H.element([8, 9])

    # 1-norm
    HxH = odl.ProductSpace(H, H, exponent=1.0)
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(HxH.dist(w1, w2),
                        H.dist(v11, v21) + H.dist(v12, v22))

    # 2-norm
    HxH = odl.ProductSpace(H, H, exponent=2.0)
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(
        HxH.dist(w1, w2),
        (H.dist(v11, v21) ** 2 + H.dist(v12, v22) ** 2) ** (1 / 2.0))

    # inf norm
    HxH = odl.ProductSpace(H, H, exponent=float('inf'))
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(
        HxH.dist(w1, w2),
        max(H.dist(v11, v21), H.dist(v12, v22)))


def test_norm():
    H = odl.rn(2)
    v1 = H.element([1, 2])
    v2 = H.element([5, 3])

    # 1-norm
    HxH = odl.ProductSpace(H, H, exponent=1.0)
    w = HxH.element([v1, v2])
    assert almost_equal(HxH.norm(w), H.norm(v1) + H.norm(v2))

    # 2-norm
    HxH = odl.ProductSpace(H, H, exponent=2.0)
    w = HxH.element([v1, v2])
    assert almost_equal(
        HxH.norm(w), (H.norm(v1) ** 2 + H.norm(v2) ** 2) ** (1 / 2.0))

    # inf norm
    HxH = odl.ProductSpace(H, H, exponent=float('inf'))
    w = HxH.element([v1, v2])
    assert almost_equal(HxH.norm(w), max(H.norm(v1), H.norm(v2)))


def test_inner():
    H = odl.rn(2)
    v1 = H.element([1, 2])
    v2 = H.element([5, 3])

    u1 = H.element([2, 3])
    u2 = H.element([6, 4])

    HxH = odl.ProductSpace(H, H)
    v = HxH.element([v1, v2])
    u = HxH.element([u1, u2])
    assert almost_equal(HxH.inner(v, u), H.inner(v1, u1) + H.inner(v2, u2))


def test_vector_weighting(exponent):
    r2 = odl.rn(2)
    r2x = r2.element([1, -1])
    r2y = r2.element([-2, 3])
    # inner = -5, dist = 5, norms = (sqrt(2), sqrt(13))

    r3 = odl.rn(3)
    r3x = r3.element([3, 4, 4])
    r3y = r3.element([1, -2, 1])
    # inner = -1, dist = 7, norms = (sqrt(41), sqrt(6))

    inners = [-5, -1]
    norms_x = [np.sqrt(2), np.sqrt(41)]
    dists = [5, 7]

    weight = [0.5, 1.5]
    pspace = odl.ProductSpace(r2, r3, weighting=weight, exponent=exponent)
    x = pspace.element((r2x, r3x))
    y = pspace.element((r2y, r3y))

    if exponent == 2.0:
        true_inner = np.sum(np.multiply(inners, weight))
        assert all_almost_equal(x.inner(y), true_inner)

    if exponent == float('inf'):
        true_norm_x = np.linalg.norm(
            np.multiply(norms_x, weight), ord=exponent)
    else:
        true_norm_x = np.linalg.norm(
            np.multiply(norms_x, np.power(weight, 1 / exponent)),
            ord=exponent)

    assert all_almost_equal(x.norm(), true_norm_x)

    if exponent == float('inf'):
        true_dist = np.linalg.norm(
            np.multiply(dists, weight), ord=exponent)
    else:
        true_dist = np.linalg.norm(
            np.multiply(dists, np.power(weight, 1 / exponent)),
            ord=exponent)
    assert all_almost_equal(x.dist(y), true_dist)


def test_const_weighting(exponent):
    r2 = odl.rn(2)
    r2x = r2.element([1, -1])
    r2y = r2.element([-2, 3])
    # inner = -5, dist = 5, norms = (sqrt(2), sqrt(13))

    r3 = odl.rn(3)
    r3x = r3.element([3, 4, 4])
    r3y = r3.element([1, -2, 1])
    # inner = -1, dist = 7, norms = (sqrt(41), sqrt(6))

    inners = [-5, -1]
    norms_x = [np.sqrt(2), np.sqrt(41)]
    dists = [5, 7]

    weight = 2.0
    pspace = odl.ProductSpace(r2, r3, weighting=weight, exponent=exponent)
    x = pspace.element((r2x, r3x))
    y = pspace.element((r2y, r3y))

    if exponent == 2.0:
        true_inner = weight * np.sum(inners)
        assert all_almost_equal(x.inner(y), true_inner)

    if exponent == float('inf'):
        true_norm_x = weight * np.linalg.norm(norms_x, ord=exponent)
    else:
        true_norm_x = (weight ** (1 / exponent) *
                       np.linalg.norm(norms_x, ord=exponent))

    assert all_almost_equal(x.norm(), true_norm_x)

    if exponent == float('inf'):
        true_dist = weight * np.linalg.norm(dists, ord=exponent)
    else:
        true_dist = (weight ** (1 / exponent) *
                     np.linalg.norm(dists, ord=exponent))

    assert all_almost_equal(x.dist(y), true_dist)


def custom_inner(x1, x2):
    inners = np.fromiter(
        (x1p.inner(x2p) for x1p, x2p in zip(x1.parts, x2.parts)),
        dtype=x1.space[0].dtype, count=len(x1))

    return x1.space.field.element(np.sum(inners))


def custom_norm(x):
    norms = np.fromiter(
        (xp.norm() for xp in x.parts),
        dtype=x.space[0].dtype, count=len(x))

    return float(np.linalg.norm(norms, ord=1))


def custom_dist(x1, x2):
    dists = np.fromiter(
        (x1p.dist(x2p) for x1p, x2p in zip(x1.parts, x2.parts)),
        dtype=x1.space[0].dtype, count=len(x1))

    return float(np.linalg.norm(dists, ord=1))


def test_custom_funcs():
    # Checking the standard 1-norm and standard inner product, just to
    # see that the functions are handled correctly.

    r2 = odl.rn(2)
    r2x = r2.element([1, -1])
    r2y = r2.element([-2, 3])
    # inner = -5, dist = 5, norms = (sqrt(2), sqrt(13))

    r3 = odl.rn(3)
    r3x = r3.element([3, 4, 4])
    r3y = r3.element([1, -2, 1])
    # inner = -1, dist = 7, norms = (sqrt(41), sqrt(6))

    pspace_2 = odl.ProductSpace(r2, r3, exponent=2.0)
    x = pspace_2.element((r2x, r3x))
    y = pspace_2.element((r2y, r3y))

    pspace_custom = odl.ProductSpace(r2, r3, inner=custom_inner)
    xc = pspace_custom.element((r2x, r3x))
    yc = pspace_custom.element((r2y, r3y))
    assert almost_equal(x.inner(y), xc.inner(yc))

    pspace_1 = odl.ProductSpace(r2, r3, exponent=1.0)
    x = pspace_1.element((r2x, r3x))
    y = pspace_1.element((r2y, r3y))

    pspace_custom = odl.ProductSpace(r2, r3, norm=custom_norm)
    xc = pspace_custom.element((r2x, r3x))
    assert almost_equal(x.norm(), xc.norm())

    pspace_custom = odl.ProductSpace(r2, r3, dist=custom_dist)
    xc = pspace_custom.element((r2x, r3x))
    yc = pspace_custom.element((r2y, r3y))
    assert almost_equal(x.dist(y), xc.dist(yc))

    with pytest.raises(TypeError):
        odl.ProductSpace(r2, r3, a=1)  # extra keyword argument

    with pytest.raises(ValueError):
        odl.ProductSpace(r2, r3, norm=custom_norm, inner=custom_inner)

    with pytest.raises(ValueError):
        odl.ProductSpace(r2, r3, dist=custom_dist, inner=custom_inner)

    with pytest.raises(ValueError):
        odl.ProductSpace(r2, r3, norm=custom_norm, dist=custom_dist)

    with pytest.raises(ValueError):
        odl.ProductSpace(r2, r3, norm=custom_norm, exponent=1.0)

    with pytest.raises(ValueError):
        odl.ProductSpace(r2, r3, norm=custom_norm, weighting=2.0)

    with pytest.raises(ValueError):
        odl.ProductSpace(r2, r3, dist=custom_dist, weighting=2.0)

    with pytest.raises(ValueError):
        odl.ProductSpace(r2, r3, inner=custom_inner, weighting=2.0)


def test_power_RxR():
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, 2)
    assert len(HxH) == 2

    v1 = H.element([1, 2])
    v2 = H.element([3, 4])
    v = HxH.element([v1, v2])
    u = HxH.element([[1, 2], [3, 4]])

    assert all_equal([v1, v2], v)
    assert all_equal([v1, v2], u)


def _test_shape(space, expected_shape):
    """Helper to validate shape of space."""
    space_el = space.element()

    assert space.shape == expected_shape
    assert space_el.shape == expected_shape
    try:
        arr = space_el.asarray()
    except ValueError:
        pass  # could not convert to array
    else:
        assert arr.shape == expected_shape
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
    _test_shape(r2_3, (3, 2))

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


def test_element_equals():
    H = odl.ProductSpace(odl.rn(1), odl.rn(2))
    x = H.element([[0], [1, 2]])

    assert x != 0  # test == not always true
    assert x == x

    x_2 = H.element([[0], [1, 2]])
    assert x == x_2

    x_3 = H.element([[3], [1, 2]])
    assert x != x_3

    x_4 = H.element([[0], [1, 3]])
    assert x != x_4


def test_element_getitem_int():
    """Test indexing of product space elements with one or several integers."""
    pspace = odl.ProductSpace(odl.rn(1), odl.rn(2))

    # One level of product space
    x0 = pspace[0].element([0])
    x1 = pspace[1].element([1, 2])
    x = pspace.element([x0, x1])

    assert x[0] is x0
    assert x[1] is x1
    assert x[-2] is x0
    assert x[-1] is x1
    with pytest.raises(IndexError):
        x[-3]
        x[2]
    assert x[0, 0] == 0
    assert x[1, 0] == 1

    # Two levels of product spaces
    pspace2 = odl.ProductSpace(pspace, 3)
    z = pspace2.element([x, x, x])
    assert z[0] is x
    assert z[1, 0] is x0
    assert z[1, 1, 1] == 2


def test_element_getitem_slice():
    """Test indexing of product space elements with slices."""
    # One level of product space
    pspace = odl.ProductSpace(odl.rn(1), odl.rn(2), odl.rn(3))

    x0 = pspace[0].element([0])
    x1 = pspace[1].element([1, 2])
    x2 = pspace[2].element([3, 4, 5])
    x = pspace.element([x0, x1, x2])

    assert x[:2].space == pspace[:2]
    assert x[:2][0] is x0
    assert x[:2][1] is x1


def test_element_getitem_fancy():
    pspace = odl.ProductSpace(odl.rn(1), odl.rn(2), odl.rn(3))

    x0 = pspace[0].element([0])
    x1 = pspace[1].element([1, 2])
    x2 = pspace[2].element([3, 4, 5])
    x = pspace.element([x0, x1, x2])

    assert x[[0, 2]].space == pspace[[0, 2]]
    assert x[[0, 2]][0] is x0
    assert x[[0, 2]][1] is x2


def test_element_getitem_multi():
    """Test element access with multiple indices."""
    pspace = odl.ProductSpace(odl.rn(1), odl.rn(2))
    pspace2 = odl.ProductSpace(pspace, 3)
    pspace3 = odl.ProductSpace(pspace2, 2)
    z = pspace3.element(
        [[[[1],
           [2, 3]],
          [[4],
           [5, 6]],
          [[7],
           [8, 9]]],
         [[[10],
           [12, 13]],
          [[14],
           [15, 16]],
          [[17],
           [18, 19]]]
         ]
    )

    assert pspace3.shape == (2, 3, 2)
    assert z[0, 0, 0, 0] == 1
    assert all_equal(z[0, 0, 1], [2, 3])
    assert all_equal(z[0, 0], [[1], [2, 3]])
    assert all_equal(z[0, 1:], [[[4],
                                 [5, 6]],
                                [[7],
                                 [8, 9]]])
    assert all_equal(z[0, 1:, 1], [[5, 6],
                                   [8, 9]])
    assert all_equal(z[0, 1:, :, 0], [[[4],
                                       [5]],
                                      [[7],
                                       [8]]])


def test_element_setitem_single():
    """Test assignment of pspace parts with single indices."""
    pspace = odl.ProductSpace(odl.rn(1), odl.rn(2))

    x0 = pspace[0].element([0])
    x1 = pspace[1].element([1, 2])
    x = pspace.element([x0, x1])
    old_x0 = x[0]
    old_x1 = x[1]

    # Check that values are set, but identity is preserved
    new_x0 = pspace[0].element([1])
    x[-2] = new_x0
    assert x[-2] == new_x0
    assert x[-2] is old_x0

    new_x1 = pspace[1].element([3, 4])
    x[-1] = new_x1
    assert x[-1] == new_x1
    assert x[-1] is old_x1

    # Set values with scalars
    x[1] = -1
    assert all_equal(x[1], [-1, -1])
    assert x[1] is old_x1

    # Check that out-of-bounds indices raise IndexError
    with pytest.raises(IndexError):
        x[-3] = x1
    with pytest.raises(IndexError):
        x[2] = x0


def test_element_setitem_slice():
    """Test assignment of pspace parts with slices."""
    pspace = odl.ProductSpace(odl.rn(1), odl.rn(2), odl.rn(3))

    x0 = pspace[0].element([0])
    x1 = pspace[1].element([1, 2])
    x2 = pspace[2].element([3, 4, 5])
    x = pspace.element([x0, x1, x2])
    old_x0 = x[0]
    old_x1 = x[1]

    # Check that values are set, but identity is preserved
    new_x0 = pspace[0].element([6])
    new_x1 = pspace[1].element([7, 8])
    x[:2] = pspace[:2].element([new_x0, new_x1])
    assert x[:2][0] is old_x0
    assert x[:2][0] == new_x0
    assert x[:2][1] is old_x1
    assert x[:2][1] == new_x1

    # Set values with sequences of scalars
    x[:2] = [-1, -2]
    assert x[:2][0] is old_x0
    assert all_equal(x[:2][0], [-1])
    assert x[:2][1] is old_x1
    assert all_equal(x[:2][1], [-2, -2])


def test_element_setitem_fancy():
    """Test assignment of pspace parts with lists."""
    pspace = odl.ProductSpace(odl.rn(1), odl.rn(2), odl.rn(3))

    x0 = pspace[0].element([0])
    x1 = pspace[1].element([1, 2])
    x2 = pspace[2].element([3, 4, 5])
    x = pspace.element([x0, x1, x2])
    old_x0 = x[0]
    old_x2 = x[2]

    # Check that values are set, but identity is preserved
    new_x0 = pspace[0].element([6])
    new_x2 = pspace[2].element([7, 8, 9])
    x[[0, 2]] = pspace[[0, 2]].element([new_x0, new_x2])
    assert x[[0, 2]][0] is old_x0
    assert x[[0, 2]][0] == new_x0
    assert x[[0, 2]][1] is old_x2
    assert x[[0, 2]][1] == new_x2

    # Set values with sequences of scalars
    x[[0, 2]] = [-1, -2]
    assert x[[0, 2]][0] is old_x0
    assert all_equal(x[[0, 2]][0], [-1])
    assert x[[0, 2]][1] is old_x2
    assert all_equal(x[[0, 2]][1], [-2, -2, -2])


def test_element_setitem_broadcast():
    """Test assignment of power space parts with broadcasting."""
    pspace = odl.ProductSpace(odl.rn(2), 3)
    x0 = pspace[0].element([0, 1])
    x1 = pspace[1].element([2, 3])
    x2 = pspace[2].element([4, 5])
    x = pspace.element([x0, x1, x2])
    old_x0 = x[0]
    old_x1 = x[1]

    # Set values with a single base space element
    new_x0 = pspace[0].element([4, 5])
    x[:2] = new_x0
    assert x[0] is old_x0
    assert x[0] == new_x0
    assert x[1] is old_x1
    assert x[1] == new_x0


def test_unary_ops():
    # Verify that the unary operators (`+x` and `-x`) work as expected

    space = odl.rn(3)
    pspace = odl.ProductSpace(space, 2)

    for op in [operator.pos, operator.neg]:
        x_arr, x = noise_elements(pspace)

        y_arr = op(x_arr)
        y = op(x)

        assert all_almost_equal([x, y], [x_arr, y_arr])


def test_operators(odl_arithmetic_op):
    # Test of the operators `+`, `-`, etc work as expected by numpy
    op = odl_arithmetic_op

    space = odl.rn(3)
    pspace = odl.ProductSpace(space, 2)

    # Interactions with scalars

    for scalar in [-31.2, -1, 0, 1, 2.13]:

        # Left op
        x_arr, x = noise_elements(pspace)
        if scalar == 0 and op in [operator.truediv, operator.itruediv]:
            # Check for correct zero division behaviour
            with pytest.raises(ZeroDivisionError):
                y = op(x, scalar)
        else:
            y_arr = op(x_arr, scalar)
            y = op(x, scalar)

            assert all_almost_equal([x, y], [x_arr, y_arr])

        # Right op
        x_arr, x = noise_elements(pspace)

        y_arr = op(scalar, x_arr)
        y = op(scalar, x)

        assert all_almost_equal([x, y], [x_arr, y_arr])

    # Verify that the statement z=op(x, y) gives equivalent results to NumPy
    x_arr, x = noise_elements(space, 1)
    y_arr, y = noise_elements(pspace, 1)

    # non-aliased left
    if op in [operator.iadd, operator.isub, operator.itruediv, operator.imul]:
        # Check for correct error since in-place op is not possible here
        with pytest.raises(TypeError):
            z = op(x, y)
    else:
        z_arr = op(x_arr, y_arr)
        z = op(x, y)

        assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # non-aliased right
    z_arr = op(y_arr, x_arr)
    z = op(y, x)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])

    # aliased operation
    z_arr = op(y_arr, y_arr)
    z = op(y, y)

    assert all_almost_equal([x, y, z], [x_arr, y_arr, z_arr])


def test_ufuncs():
    # Cannot use fixture due to bug in pytest
    H = odl.ProductSpace(odl.rn(1), odl.rn(2))

    # one arg
    x = H.element([[-1], [-2, -3]])

    z = x.ufuncs.absolute()
    assert all_almost_equal(z, [[1], [2, 3]])

    # one arg with out
    x = H.element([[-1], [-2, -3]])
    y = H.element()

    z = x.ufuncs.absolute(out=y)
    assert y is z
    assert all_almost_equal(z, [[1], [2, 3]])

    # Two args
    x = H.element([[1], [2, 3]])
    y = H.element([[4], [5, 6]])
    w = H.element()

    z = x.ufuncs.add(y)
    assert all_almost_equal(z, [[5], [7, 9]])

    # Two args with out
    x = H.element([[1], [2, 3]])
    y = H.element([[4], [5, 6]])
    w = H.element()

    z = x.ufuncs.add(y, out=w)
    assert w is z
    assert all_almost_equal(z, [[5], [7, 9]])


def test_reductions():
    H = odl.ProductSpace(odl.rn(1), odl.rn(2))
    x = H.element([[1], [2, 3]])
    assert x.ufuncs.sum() == 6.0
    assert x.ufuncs.prod() == 6.0
    assert x.ufuncs.min() == 1.0
    assert x.ufuncs.max() == 3.0


def test_array_wrap_method():
    """Verify that the __array_wrap__ method for NumPy works."""
    space = odl.ProductSpace(odl.uniform_discr(0, 1, 10), 2)
    x_arr, x = noise_elements(space)
    y_arr = np.sin(x_arr)
    y = np.sin(x)  # Should yield again an ODL product space element

    assert y in space
    assert all_equal(y, y_arr)


def test_real_imag_and_conj():
    """Verify that .real .imag and .conj() work for product space elements."""
    space = odl.ProductSpace(odl.uniform_discr(0, 1, 3, dtype=complex),
                             odl.cn(2))
    x = noise_element(space)

    # Test real
    expected_result = space.real_space.element([x[0].real, x[1].real])
    assert x.real == expected_result

    # Test imag
    expected_result = space.real_space.element([x[0].imag, x[1].imag])
    assert x.imag == expected_result

    # Test conj. Note that ProductSpace does not implement asarray if
    # is_power_space is false. Hence the construction below
    expected_result = space.element([x[0].conj(), x[1].conj()])
    x_conj = x.conj()
    assert x_conj[0] == expected_result[0]
    assert x_conj[1] == expected_result[1]


def test_real_setter_product_space(space, newpart):
    """Verify that the setter for the real part of an element works."""
    x = noise_element(space)
    x.real = newpart

    try:
        # Catch the scalar
        iter(newpart)
    except TypeError:
        expected_result = newpart * space.one()
    else:
        if newpart in space:
            expected_result = newpart.real
        elif np.shape(newpart) == (3,):
            expected_result = [newpart, newpart]
        else:
            expected_result = newpart

    assert x in space
    assert all_equal(x.real, expected_result)


def test_imag_setter_product_space(space, newpart):
    """Verify that the setter for the imaginary part of an element works."""
    x = noise_element(space)
    x.imag = newpart

    try:
        # Catch the scalar
        iter(newpart)
    except TypeError:
        expected_result = newpart * space.one()
    else:
        if newpart in space:
            # The imaginary part is by definition real, and thus the new
            # imaginary part is thus the real part of the element we try to set
            # the value to
            expected_result = newpart.real
        elif np.shape(newpart) == (3,):
            expected_result = [newpart, newpart]
        else:
            expected_result = newpart

    assert x in space
    assert all_equal(x.imag, expected_result)


if __name__ == '__main__':
    odl.util.test_file(__file__)
