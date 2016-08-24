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

# ODL imports
import odl
from odl.util.testutils import all_almost_equal


def test_pspace_op_init():
    r3 = odl.rn(3)
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


def test_pspace_op_weighted_init():

    r3 = odl.rn(3)
    ran = odl.ProductSpace(r3, 2, weight=[1, 2])
    I = odl.IdentityOperator(r3)

    with pytest.raises(NotImplementedError):
        odl.ProductSpaceOperator([[I],
                                  [0]], range=ran)


def test_pspace_op_sum_call():
    r3 = odl.rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([I, I])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert all_almost_equal(op(z)[0], x + y)
    assert all_almost_equal(op(z, out=op.range.element())[0], x + y)


def test_pspace_op_project_call():
    r3 = odl.rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[I],
                                   [I]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert x == op(z)[0]
    assert x == op(z, out=op.range.element())[0]


def test_pspace_op_diagonal_call():
    r3 = odl.rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[I, 0],
                                   [0, I]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert z == op(z)
    assert z == op(z, out=op.range.element())


def test_pspace_op_swap_call():
    r3 = odl.rn(3)
    I = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[0, I],
                                   [I, 0]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])
    result = op.domain.element([y, x])

    assert result == op(z)
    assert result == op(z, out=op.range.element())


def test_comp_proj():
    r3 = odl.rn(3)
    r3xr3 = odl.ProductSpace(r3, 2)

    x = r3xr3.element([[1, 2, 3],
                       [4, 5, 6]])
    proj_0 = odl.ComponentProjection(r3xr3, 0)
    assert x[0] == proj_0(x)
    assert x[0] == proj_0(x, out=proj_0.range.element())

    proj_1 = odl.ComponentProjection(r3xr3, 1)
    assert x[1] == proj_1(x)
    assert x[1] == proj_1(x, out=proj_1.range.element())


def test_comp_proj_slice():
    r3 = odl.rn(3)
    r33 = odl.ProductSpace(r3, 3)

    x = r33.element([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    proj = odl.ComponentProjection(r33, slice(0, 2))
    assert x[0:2] == proj(x)
    assert x[0:2] == proj(x, out=proj.range.element())


def test_comp_proj_indices():
    r3 = odl.rn(3)
    r33 = odl.ProductSpace(r3, 3)

    x = r33.element([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    proj = odl.ComponentProjection(r33, [0, 2])
    assert x[[0, 2]] == proj(x)
    assert x[[0, 2]] == proj(x, out=proj.range.element())


def test_comp_proj_adjoint():
    r3 = odl.rn(3)
    r3xr3 = odl.ProductSpace(r3, 2)

    x = r3.element([1, 2, 3])

    result_0 = r3xr3.element([[1, 2, 3],
                              [0, 0, 0]])
    proj_0 = odl.ComponentProjection(r3xr3, 0)

    assert result_0 == proj_0.adjoint(x)
    assert result_0 == proj_0.adjoint(x, out=proj_0.domain.element())

    result_1 = r3xr3.element([[0, 0, 0],
                              [1, 2, 3]])
    proj_1 = odl.ComponentProjection(r3xr3, 1)

    assert result_1 == proj_1.adjoint(x)
    assert result_1 == proj_1.adjoint(x, out=proj_1.domain.element())


def test_comp_proj_adjoint_slice():
    r3 = odl.rn(3)
    r33 = odl.ProductSpace(r3, 3)

    x = r33[0:2].element([[1, 2, 3],
                          [4, 5, 6]])

    result = r33.element([[1, 2, 3],
                          [4, 5, 6],
                          [0, 0, 0]])
    proj = odl.ComponentProjection(r33, slice(0, 2))

    assert result == proj.adjoint(x)
    assert result == proj.adjoint(x, out=proj.domain.element())


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
