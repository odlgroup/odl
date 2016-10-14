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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External module imports
import pytest
import numpy as np

# ODL imports
from odl.discr.grid import sparse_meshgrid
from odl.set.domain import IntervalProd
from odl.util.testutils import almost_equal, all_equal


def random_point(set_):
    if isinstance(set_, IntervalProd):
        return (set_.min_pt +
                np.random.rand(set_.ndim) * (set_.max_pt - set_.min_pt))
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
    set_ = IntervalProd(1, 2)
    assert almost_equal(set_.min_pt, 1)

    set_ = IntervalProd(-np.inf, 0)
    assert almost_equal(set_.min_pt, -np.inf)

    set_ = IntervalProd([1], [2])
    assert almost_equal(set_.min_pt, 1)

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_equal(set_.min_pt, [1, 2, 3])


def test_max_pt():
    set_ = IntervalProd(1, 2)
    assert almost_equal(set_.max_pt, 2)

    set_ = IntervalProd(0, np.inf)
    assert almost_equal(set_.max_pt, np.inf)

    set_ = IntervalProd([1], [2])
    assert almost_equal(set_.max_pt, 2)

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_equal(set_.max_pt, [5, 6, 7])


def test_ndim():
    set_ = IntervalProd(1, 2)
    assert set_.ndim == 1

    set_ = IntervalProd(1, 1)
    assert set_.ndim == 1

    set_ = IntervalProd(0, np.inf)
    assert set_.ndim == 1

    set_ = IntervalProd([1], [2])
    assert set_.ndim == 1

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert set_.ndim == 3

    set_ = IntervalProd([1, 2, 3], [1, 6, 7])
    assert set_.ndim == 3


def test_true_ndim():
    set_ = IntervalProd(1, 2)
    assert set_.true_ndim == 1

    set_ = IntervalProd(1, 1)
    assert set_.true_ndim == 0

    set_ = IntervalProd(0, np.inf)
    assert set_.true_ndim == 1

    set_ = IntervalProd([1], [2])
    assert set_.true_ndim == 1

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert set_.true_ndim == 3

    set_ = IntervalProd([1, 2, 3], [1, 6, 7])
    assert set_.true_ndim == 2


def test_extent():
    set_ = IntervalProd(1, 2)
    assert set_.extent() == 1

    set_ = IntervalProd(1, 1)
    assert set_.extent() == 0

    set_ = IntervalProd(0, np.inf)
    assert set_.extent() == np.inf

    set_ = IntervalProd(-np.inf, 0)
    assert set_.extent() == np.inf

    set_ = IntervalProd(-np.inf, np.inf)
    assert set_.extent() == np.inf

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert list(set_.extent()) == [4, 4, 4]


def test_volume():
    set_ = IntervalProd(1, 2)
    assert set_.volume == 2 - 1

    set_ = IntervalProd(0, np.inf)
    assert set_.volume == np.inf

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert almost_equal(set_.volume, (5 - 1) * (6 - 2) * (7 - 3))


def test_mid_pt():
    set_ = IntervalProd(1, 2)
    assert set_.mid_pt == 1.5

    set_ = IntervalProd(0, np.inf)
    assert set_.mid_pt == np.inf

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_equal(set_.mid_pt, [3, 4, 5])


def test_element():
    set_ = IntervalProd(1, 2)
    assert set_.element() in set_

    set_ = IntervalProd(0, np.inf)
    assert set_.element() in set_

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert set_.element() in set_


def test_equals():
    interval1 = IntervalProd(1, 2)
    interval2 = IntervalProd(1, 2)
    interval3 = IntervalProd([1], [2])
    interval4 = IntervalProd(2, 3)
    rectangle1 = IntervalProd([1, 2], [2, 3])
    rectangle2 = IntervalProd((1, 2), (2, 3))
    rectangle3 = IntervalProd([0, 2], [2, 3])

    assert interval1 == interval1
    assert not interval1 != interval1
    assert interval1 == interval2
    assert interval1 == interval3
    assert not interval1 == interval4
    assert interval1 != interval4
    assert not interval1 == rectangle1
    assert rectangle1 == rectangle1
    assert rectangle2 == rectangle2
    assert not rectangle1 == rectangle3

    r1_1 = IntervalProd(-np.inf, np.inf)
    r1_2 = IntervalProd(-np.inf, np.inf)
    positive_reals = IntervalProd(0, np.inf)
    assert r1_1 == r1_1
    assert r1_1 == r1_2

    assert positive_reals == positive_reals
    assert positive_reals != r1_1

    # non interval
    for non_interval in [1, 1j, np.array([1, 2])]:
        assert interval1 != non_interval
        assert rectangle1 != non_interval


def test_contains():
    set_ = IntervalProd(1, 2)

    assert 1 in set_
    assert 2 in set_
    assert 1.5 in set_
    assert 3 not in set_
    assert 'string' not in set_
    assert [1, 2] not in set_
    assert np.nan not in set_

    positive_reals = IntervalProd(0, np.inf)
    assert 1 in positive_reals
    assert np.inf in positive_reals
    assert -np.inf not in positive_reals
    assert -1 not in positive_reals


def test_contains_set():
    set_ = IntervalProd(1, 2)

    for sub_set in [np.array([1, 1.1, 1.2, 1.3, 1.4]),
                    IntervalProd(1.2, 2),
                    IntervalProd(1, 1.5),
                    IntervalProd(1.2, 1.2)]:
        assert set_.contains_set(sub_set)

    for non_sub_set in [np.array([0, 1, 1.1, 1.2, 1.3, 1.4]),
                        np.array([np.nan, 1.1, 1.3]),
                        IntervalProd(1.2, 3),
                        IntervalProd(0, 1.5),
                        IntervalProd(3, 4)]:
        assert not set_.contains_set(non_sub_set)

    for non_set in [1,
                    [1, 2],
                    {'hello': 1.0}]:
        with pytest.raises(AttributeError):
            set_.contains_set(non_set)


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
    set_ = IntervalProd(1, 2)

    for interior in [1.0, 1.1, 2.0]:
        assert set_.dist(interior) == 0.0

    for exterior in [0.0, 2.0, np.inf]:
        assert set_.dist(exterior) == min(abs(set_.min_pt - exterior),
                                          abs(exterior - set_.max_pt))

    assert set_.dist(np.NaN) == np.inf


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
    set_ = IntervalProd(1, 2)
    assert set_.length == set_.volume
    assert set_.length == 1


def test_rectangle_area():
    set_ = IntervalProd([1, 2], [3, 4])
    assert set_.area == set_.volume
    assert set_.area == (3 - 1) * (4 - 2)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
