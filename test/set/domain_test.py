# Copyright 2014, 2015 The ODL development group
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
from odl.set.domain import IntervalProd, Interval, Rectangle
from odl.util.testutils import almost_equal, all_almost_equal


# TODO:
# - Interval arithmetics


def random_point(set_):
    if isinstance(set_, IntervalProd):
        return np.random.rand(set_.ndim) * (set_.end - set_.begin) + set_.begin
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

def test_begin():
    set_ = IntervalProd(1, 2)
    assert almost_equal(set_.begin, 1)

    set_ = IntervalProd(-np.inf, 0)
    assert almost_equal(set_.begin, -np.inf)

    set_ = IntervalProd([1], [2])
    assert almost_equal(set_.begin, 1)

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_almost_equal(set_.begin, [1, 2, 3])

def test_end():
    set_ = IntervalProd(1, 2)
    assert almost_equal(set_.end, 2)

    set_ = IntervalProd(0, np.inf)
    assert almost_equal(set_.end, np.inf)

    set_ = IntervalProd([1], [2])
    assert almost_equal(set_.end, 2)

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert all_almost_equal(set_.end, [5, 6, 7])

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

def test_size():
    set_ = IntervalProd(1, 2)
    assert set_.size == 1

    set_ = IntervalProd(1, 1)
    assert set_.size == 0

    set_ = IntervalProd(0, np.inf)
    assert set_.size == np.inf

    set_ = IntervalProd(-np.inf, 0)
    assert set_.size == np.inf

    set_ = IntervalProd(-np.inf, np.inf)
    assert set_.size == np.inf

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert list(set_.size) == [4, 4, 4]

def test_volume():
    set_ = IntervalProd(1, 2)
    assert set_.volume == 2-1

    set_ = IntervalProd(0, np.inf)
    assert set_.volume == np.inf

    set_ = IntervalProd([1, 2, 3], [5, 6, 7])
    assert almost_equal(set_.volume, (5-1)*(6-2)*(7-3))

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

def test_contains():
    set_ = IntervalProd(1, 2)

    assert 1 in set_
    assert 2 in set_
    assert 1.5 in set_
    assert not 3 in set_
    assert 3 not in set_

    positive_reals = IntervalProd(0, np.inf)
    assert 1 in positive_reals
    assert np.inf in positive_reals
    assert not -1 in positive_reals


def test_interval_init():
    Interval(1, 2)
    Interval([1], [2])

    with pytest.raises(ValueError):
        Interval([1, 2], [3, 4])

def test_interval_length():
    set_ = Interval(1, 2)
    assert set_.length == set_.volume
    assert set_.length == 1


def test_rectangle_init():
    Rectangle([1, 2], [2, 3])

    with pytest.raises(ValueError):
        Rectangle(1, 2)

    with pytest.raises(ValueError):
        Rectangle([1, 2, 3], [4, 5, 6])

def test_rectangle_area():
    set_ = Rectangle([1, 2], [3, 4])
    assert set_.area == set_.volume
    assert set_.area == (3-1)*(4-2)

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\','/') + ' -v'))