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
from builtins import str

# External module imports
import numpy as np
import pytest

# ODL imports
import odl
from odl.util.testutils import all_equal, all_almost_equal, almost_equal


def test_RxR():
    H = odl.Rn(2)
    HxH = odl.ProductSpace(H, H)
    assert len(HxH) == 2

    v1 = H.element([1, 2])
    v2 = H.element([3, 4])
    v = HxH.element([v1, v2])
    u = HxH.element([[1, 2], [3, 4]])

    assert all_equal([v1, v2], v)
    assert all_equal([v1, v2], u)


def test_lincomb():
    H = odl.Rn(2)
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
    H = odl.Rn(2)
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
    H = odl.Rn(2)
    v11 = H.element([1, 2])
    v12 = H.element([5, 3])

    v21 = H.element([1, 2])
    v22 = H.element([8, 9])

    # 1-norm
    HxH = odl.ProductSpace(H, H, ord=1.0)
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(HxH.dist(w1, w2),
                        H.dist(v11, v21) + H.dist(v12, v22))

    # 2-norm
    HxH = odl.ProductSpace(H, H, ord=2.0)
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(
        HxH.dist(w1, w2),
        (H.dist(v11, v21) ** 2 + H.dist(v12, v22) ** 2) ** (1 / 2.0))

    # -inf norm
    HxH = odl.ProductSpace(H, H, ord=-float('inf'))
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(
        HxH.dist(w1, w2),
        min(H.dist(v11, v21), H.dist(v12, v22)))

    # inf norm
    HxH = odl.ProductSpace(H, H, ord=float('inf'))
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(
        HxH.dist(w1, w2),
        max(H.dist(v11, v21), H.dist(v12, v22)))

    # Custom norm
    def my_norm(x):
        return np.sum(x)  # Same as 1-norm
    HxH = odl.ProductSpace(H, H, prod_dist=my_norm)
    w1 = HxH.element([v11, v12])
    w2 = HxH.element([v21, v22])
    assert almost_equal(
        HxH.dist(w1, w2),
        H.dist(v11, v21) + H.dist(v12, v22))


def test_norm():
    H = odl.Rn(2)
    v1 = H.element([1, 2])
    v2 = H.element([5, 3])

    # 1-norm
    HxH = odl.ProductSpace(H, H, ord=1.0)
    w = HxH.element([v1, v2])
    assert almost_equal(HxH.norm(w), H.norm(v1) + H.norm(v2))

    # 2-norm
    HxH = odl.ProductSpace(H, H, ord=2.0)
    w = HxH.element([v1, v2])
    assert almost_equal(
        HxH.norm(w), (H.norm(v1) ** 2 + H.norm(v2) ** 2) ** (1 / 2.0))

    # -inf norm
    HxH = odl.ProductSpace(H, H, ord=-float('inf'))
    w = HxH.element([v1, v2])
    assert almost_equal(HxH.norm(w), min(H.norm(v1), H.norm(v2)))

    # inf norm
    HxH = odl.ProductSpace(H, H, ord=float('inf'))
    w = HxH.element([v1, v2])
    assert almost_equal(HxH.norm(w), max(H.norm(v1), H.norm(v2)))

    # Custom norm
    def my_norm(x):
        return np.sum(x)  # Same as 1-norm
    HxH = odl.ProductSpace(H, H, prod_norm=my_norm)
    w = HxH.element([v1, v2])
    assert almost_equal(HxH.norm(w), H.norm(v1) + H.norm(v2))


def test_inner():
    H = odl.Rn(2)
    v1 = H.element([1, 2])
    v2 = H.element([5, 3])

    u1 = H.element([2, 3])
    u2 = H.element([6, 4])

    HxH = odl.ProductSpace(H, H)
    v = HxH.element([v1, v2])
    u = HxH.element([u1, u2])
    assert almost_equal(HxH.inner(v, u), H.inner(v1, u1) + H.inner(v2, u2))


def test_power_RxR():
    H = odl.Rn(2)
    HxH = odl.ProductSpace(H, 2)
    assert len(HxH) == 2

    v1 = H.element([1, 2])
    v2 = H.element([3, 4])
    v = HxH.element([v1, v2])
    u = HxH.element([[1, 2], [3, 4]])

    assert all_equal([v1, v2], v)
    assert all_equal([v1, v2], u)


def test_power_lincomb():
    H = odl.Rn(2)
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


def test_power_inplace_modify():
    H = odl.Rn(2)
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


def test_getitem():
    H = odl.Rn(2)
    HxH = odl.ProductSpace(H, 2)

    assert HxH[-2] == H
    assert HxH[-1] == H
    assert HxH[0] == H
    assert HxH[1] == H
    with pytest.raises(IndexError):
        HxH[-3]
        HxH[2]

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
