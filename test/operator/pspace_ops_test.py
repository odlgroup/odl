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

# ODL imports
import odl
from odl.util.testutils import all_almost_equal


def test_init():
    r3 = odl.Rn(3)
    I = odl.IdentityOperator(r3)

    op = odl.ProductSpaceOperator([I])
    assert op.domain == odl.ProductSpace(r3)
    assert op.range == odl.ProductSpace(r3)

    op = odl.ProductSpaceOperator([I, I])
    assert op.domain == odl.ProductSpace(r3, 2)
    assert op.range == odl.ProductSpace(r3)

    op = odl.ProductSpaceOperator([[I],
                                   [I]])
    assert op.domain == odl.ProductSpace(r3)
    assert op.range == odl.ProductSpace(r3, 2)

    op = odl.ProductSpaceOperator([[I, 0],
                                   [0, I]])
    assert op.domain == odl.ProductSpace(r3, 2)
    assert op.range == odl.ProductSpace(r3, 2)

    op = odl.ProductSpaceOperator([[I, None],
                                   [None, I]])
    assert op.domain == odl.ProductSpace(r3, 2)
    assert op.range == odl.ProductSpace(r3, 2)


def test_sum_call():
    r3 = odl.Rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([I, I])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert all_almost_equal(op(z)[0], x + y)


def test_project_call():
    r3 = odl.Rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[I],
                                   [I]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert all_almost_equal(op(z)[0], x)


def test_diagonal_call():
    r3 = odl.Rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[I, 0],
                                   [0, I]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert all_almost_equal(op(z), z)


def test_swap_call():
    r3 = odl.Rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[0, I],
                                   [I, 0]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])
    result = op.domain.element([y, x])

    assert all_almost_equal(op(z), result)


def test_projection():
    r3 = odl.Rn(3)
    r3xr3 = odl.ProductSpace(r3, 2)

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = r3xr3.element([x, y])
    proj_0 = odl.ComponentProjection(r3xr3, 0)
    assert x == proj_0(z)

    proj_1 = odl.ComponentProjection(r3xr3, 1)
    assert y == proj_1(z)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
