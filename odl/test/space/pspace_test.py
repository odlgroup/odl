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
import operator

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
    assert pspace.dtype == 'float64'

    r2_x = r2.element([1, 2])
    r3_x = r3.element([3, 4, 5])
    x = pspace.element([r2_x, r3_x])
    y = pspace.element([[1, 2], [3, 4, 5]])
    assert all_equal(x, y)
    assert all_equal(x, [r2_x, r3_x])

    # Power space
    pspace = odl.ProductSpace(r3, 2)
    assert len(pspace) == 2
    assert pspace.shape == (2, 3)
    assert pspace.size == 6
    assert pspace.spaces[0] == pspace.spaces[1] == r3
    assert pspace.is_power_space
    assert not pspace.is_weighted
    assert pspace.is_real
    assert not pspace.is_complex
    assert pspace.dtype == 'float64'

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


def test_pspace_element():
    """Test element creation in product spaces."""
    H = odl.rn(2)
    HxH = odl.ProductSpace(H, H)
    elem = HxH.element([[1, 2], [3, 4]])
    assert elem in HxH

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


def test_np_reductions():
    """Check that reductions via NumPy functions work."""
    H = odl.ProductSpace(odl.rn(2), 3)
    x = 2 * H.one()
    assert np.sum(x) == 2 * 6
    assert np.prod(x) == 2 ** 6


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
