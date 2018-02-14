# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division
import pytest
import numpy as np

import odl
from odl.discr.grid import sparse_meshgrid
from odl.set.domain import IntervalProd
from odl.util.testutils import all_equal


def random_point(intv):
    if isinstance(intv, IntervalProd):
        return (intv.min_pt +
                np.random.rand(intv.ndim) * (intv.max_pt - intv.min_pt))
    else:
        raise NotImplementedError("unknown type")


def test_init():
    IntervalProd(1, 2)
    IntervalProd(-np.inf, 2)
    IntervalProd(0, np.inf)
    IntervalProd([1], [2])
    IntervalProd((1,), (2,))

    IntervalProd([1, 2, 3], [4, 5, 6])
    IntervalProd((1, 2, 3), (4, 5, 6))

    IntervalProd((1, 2, 3), (1, 2, 3))

    with pytest.raises(ValueError):
        IntervalProd(2, 1)

    with pytest.raises(ValueError):
        IntervalProd((1, 2, 3), (1, 2, 0))


def test_min_pt():
    intv = IntervalProd(1, 2)
    assert intv.min_pt == 1

    intv = IntervalProd(-np.inf, 0)
    assert intv.min_pt == -np.inf

    intv = IntervalProd([1], [2])
    assert intv.min_pt == 1

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_equal(intv.min_pt, [1, 2, 3])


def test_max_pt():
    intv = IntervalProd(1, 2)
    assert intv.max_pt == 2

    intv = IntervalProd(0, np.inf)
    assert intv.max_pt == np.inf

    intv = IntervalProd([1], [2])
    assert intv.max_pt == 2

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_equal(intv.max_pt, [5, 6, 7])


def test_ndim():
    intv = IntervalProd(1, 2)
    assert intv.ndim == 1

    intv = IntervalProd(1, 1)
    assert intv.ndim == 1

    intv = IntervalProd(0, np.inf)
    assert intv.ndim == 1

    intv = IntervalProd([1], [2])
    assert intv.ndim == 1

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert intv.ndim == 3

    intv = IntervalProd([1, 2, 3], [1, 6, 7])
    assert intv.ndim == 3


def test_true_ndim():
    intv = IntervalProd(1, 2)
    assert intv.true_ndim == 1

    intv = IntervalProd(1, 1)
    assert intv.true_ndim == 0

    intv = IntervalProd(0, np.inf)
    assert intv.true_ndim == 1

    intv = IntervalProd([1], [2])
    assert intv.true_ndim == 1

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert intv.true_ndim == 3

    intv = IntervalProd([1, 2, 3], [1, 6, 7])
    assert intv.true_ndim == 2


def test_extent():
    intv = IntervalProd(1, 2)
    assert intv.extent == 1

    intv = IntervalProd(1, 1)
    assert intv.extent == 0

    intv = IntervalProd(0, np.inf)
    assert intv.extent == np.inf

    intv = IntervalProd(-np.inf, 0)
    assert intv.extent == np.inf

    intv = IntervalProd(-np.inf, np.inf)
    assert intv.extent == np.inf

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert list(intv.extent) == [4, 4, 4]


def test_volume():
    intv = IntervalProd(1, 2)
    assert intv.volume == 2 - 1

    intv = IntervalProd(0, np.inf)
    assert intv.volume == np.inf

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert intv.volume == pytest.approx((5 - 1) * (6 - 2) * (7 - 3))


def test_mid_pt():
    intv = IntervalProd(1, 2)
    assert intv.mid_pt == 1.5

    intv = IntervalProd(0, np.inf)
    assert intv.mid_pt == np.inf

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_equal(intv.mid_pt, [3, 4, 5])


def test_element():
    intv = IntervalProd(1, 2)
    assert intv.element() in intv

    intv = IntervalProd(0, np.inf)
    assert intv.element() in intv

    intv = IntervalProd([1, 2, 3], [5, 6, 7])
    assert intv.element() in intv


def _test_eq(x, y):
    """Test equality of x and y."""
    assert x == y
    assert not x != y
    assert hash(x) == hash(y)


def _test_neq(x, y):
    """Test non-equality of x and y."""
    assert x != y
    assert not x == y
    assert hash(x) != hash(y)


def test_equals():
    """Test equality check of IntervalProd."""
    interval1 = IntervalProd(1, 2)
    interval2 = IntervalProd(1, 2)
    interval3 = IntervalProd([1], [2])
    interval4 = IntervalProd(2, 3)
    rectangle1 = IntervalProd([1, 2], [2, 3])
    rectangle2 = IntervalProd((1, 2), (2, 3))
    rectangle3 = IntervalProd([0, 2], [2, 3])

    _test_eq(interval1, interval1)
    _test_eq(interval1, interval2)
    _test_eq(interval1, interval3)
    _test_neq(interval1, interval4)
    _test_neq(interval1, rectangle1)
    _test_eq(rectangle1, rectangle1)
    _test_eq(rectangle2, rectangle2)
    _test_eq(rectangle1, rectangle2)
    _test_neq(rectangle1, rectangle3)

    r1_1 = IntervalProd(-np.inf, np.inf)
    r1_2 = IntervalProd(-np.inf, np.inf)
    positive_reals = IntervalProd(0, np.inf)
    _test_eq(r1_1, r1_1)
    _test_eq(r1_1, r1_2)
    _test_eq(positive_reals, positive_reals)
    _test_neq(positive_reals, r1_1)

    # non interval
    for non_interval in [1, 1j, np.array([1, 2])]:
        assert interval1 != non_interval
        assert rectangle1 != non_interval


def test_contains():
    intv = IntervalProd(1, 2)

    assert 1 in intv
    assert 2 in intv
    assert 1.5 in intv
    assert 3 not in intv
    assert 'string' not in intv
    assert [1, 2] not in intv
    assert np.nan not in intv

    positive_reals = IntervalProd(0, np.inf)
    assert 1 in positive_reals
    assert np.inf in positive_reals
    assert -np.inf not in positive_reals
    assert -1 not in positive_reals


def test_contains_set():
    intv = IntervalProd(1, 2)

    for sub_set in [np.array([1, 1.1, 1.2, 1.3, 1.4]),
                    IntervalProd(1.2, 2),
                    IntervalProd(1, 1.5),
                    IntervalProd(1.2, 1.2)]:
        assert intv.contains_set(sub_set)

    for non_sub_set in [np.array([0, 1, 1.1, 1.2, 1.3, 1.4]),
                        np.array([np.nan, 1.1, 1.3]),
                        IntervalProd(1.2, 3),
                        IntervalProd(0, 1.5),
                        IntervalProd(3, 4)]:
        assert not intv.contains_set(non_sub_set)

    for non_set in [1,
                    [1, 2],
                    {'hello': 1.0}]:
        with pytest.raises(AttributeError):
            intv.contains_set(non_set)


def test_contains_all():
    # 1d
    intvp = IntervalProd(1, 2)

    arr_in1 = np.array([1.0, 1.6, 1.3, 2.0])
    arr_in2 = np.array([[1.0, 1.6, 1.3, 2.0]])
    mesh_in = sparse_meshgrid([1.0, 1.7, 1.9])
    arr_not_in1 = np.array([1.0, 1.6, 1.3, 2.0, 0.8])
    arr_not_in2 = np.array([[1.0, 1.6, 2.0], [1.0, 1.5, 1.6]])
    mesh_not_in = sparse_meshgrid([-1.0, 1.7, 1.9])

    assert intvp.contains_all(arr_in1)
    assert intvp.contains_all(arr_in2)
    assert intvp.contains_all(mesh_in)
    assert not intvp.contains_all(arr_not_in1)
    assert not intvp.contains_all(arr_not_in2)
    assert not intvp.contains_all(mesh_not_in)

    # 2d
    intvp = IntervalProd([0, 0], [1, 2])
    arr_in = np.array([[0.5, 1.9],
                       [0.0, 1.0],
                       [1.0, 0.1]]).T
    mesh_in = sparse_meshgrid([0, 0.1, 0.6, 0.7, 1.0], [0])
    arr_not_in = np.array([[0.5, 1.9],
                           [1.1, 1.0],
                           [1.0, 0.1]]).T
    mesh_not_in = sparse_meshgrid([0, 0.1, 0.6, 0.7, 1.0], [0, -1])

    assert intvp.contains_all(arr_in)
    assert intvp.contains_all(mesh_in)
    assert not intvp.contains_all(arr_not_in)
    assert not intvp.contains_all(mesh_not_in)


def test_insert():
    intvp1 = IntervalProd([0, 0], [1, 2])
    intvp2 = IntervalProd(1, 3)

    intvp = intvp1.insert(0, intvp2)
    true_min_pt = [1, 0, 0]
    true_max_pt = [3, 1, 2]
    assert intvp == IntervalProd(true_min_pt, true_max_pt)

    intvp = intvp1.insert(1, intvp2)
    true_min_pt = [0, 1, 0]
    true_max_pt = [1, 3, 2]
    assert intvp == IntervalProd(true_min_pt, true_max_pt)

    intvp = intvp1.insert(2, intvp2)
    true_min_pt = [0, 0, 1]
    true_max_pt = [1, 2, 3]
    assert intvp == IntervalProd(true_min_pt, true_max_pt)

    intvp = intvp1.insert(-1, intvp2)  # same as 1
    true_min_pt = [0, 1, 0]
    true_max_pt = [1, 3, 2]
    assert intvp == IntervalProd(true_min_pt, true_max_pt)

    with pytest.raises(IndexError):
        intvp1.insert(3, intvp2)
    with pytest.raises(IndexError):
        intvp1.insert(-4, intvp2)


def test_dist():
    intv = IntervalProd(1, 2)

    for interior in [1.0, 1.1, 2.0]:
        assert intv.dist(interior) == 0.0

    for exterior in [0.0, 2.0, np.inf]:
        assert intv.dist(exterior) == min(abs(intv.min_pt - exterior),
                                          abs(exterior - intv.max_pt))

    assert intv.dist(np.NaN) == np.inf


# Set arithmetic
def test_pos():
    interv = IntervalProd(1, 2)
    assert +interv == interv

    interv = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert +interv == interv


def test_neg():
    interv = IntervalProd(1, 2)
    assert -interv == IntervalProd(-2, -1)

    interv = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    assert -interv == IntervalProd([-2, -3, -4, -5, -6],
                                   [-1, -2, -3, -4, -5])


def test_add():
    interv1 = IntervalProd(1, 2)
    interv2 = IntervalProd(3, 4)

    assert interv1 + 2.0 == IntervalProd(3, 4)
    assert interv1 + interv2 == IntervalProd(4, 6)

    interv1 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    interv2 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])

    assert interv1 + interv2 == IntervalProd([2, 4, 6, 8, 10],
                                             [4, 6, 8, 10, 12])


def test_sub():
    interv1 = IntervalProd(1, 2)
    interv2 = IntervalProd(3, 4)

    assert interv1 - 2.0 == IntervalProd(-1, 0)
    assert interv1 - interv2 == IntervalProd(-3, -1)

    interv1 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    interv2 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])

    assert interv1 - interv2 == IntervalProd([-1, -1, -1, -1, -1],
                                             [1, 1, 1, 1, 1])


def test_mul():
    interv1 = IntervalProd(1, 2)
    interv2 = IntervalProd(3, 4)

    assert interv1 * 2.0 == IntervalProd(2, 4)
    assert interv1 * interv2 == IntervalProd(3, 8)

    interv1 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    interv2 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])

    assert interv1 * interv2 == IntervalProd([1, 4, 9, 16, 25],
                                             [4, 9, 16, 25, 36])


def test_div():
    interv1 = IntervalProd(1, 2)
    interv2 = IntervalProd(3, 4)

    assert interv1 / 2.0 == IntervalProd(1 / 2.0, 2 / 2.0)
    assert 2.0 / interv1 == IntervalProd(2 / 2.0, 2 / 1.0)
    assert interv1 / interv2 == IntervalProd(1 / 4.0, 2.0 / 3.0)

    interv_with_zero = IntervalProd(-1, 1)
    with pytest.raises(ValueError):
        interv1 / interv_with_zero

    interv1 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    interv2 = IntervalProd([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])
    quotient = IntervalProd([1 / 2., 2 / 3., 3 / 4., 4 / 5., 5 / 6.],
                            [2 / 1., 3 / 2., 4 / 3., 5 / 4., 6 / 5.])

    assert (interv1 / interv2).approx_equals(quotient, atol=1e-10)


def test_interval_length():
    intv = IntervalProd(1, 2)
    assert intv.length == intv.volume
    assert intv.length == 1


def test_rectangle_area():
    intv = IntervalProd([1, 2], [3, 4])
    assert intv.area == intv.volume
    assert intv.area == (3 - 1) * (4 - 2)


if __name__ == '__main__':
    odl.util.test_file(__file__)
