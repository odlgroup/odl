# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import division

import pytest

import odl
from odl.util.testutils import all_almost_equal, simple_fixture

base_op = simple_fixture(
    'base_op',
    [odl.IdentityOperator(odl.rn(3)),
     odl.BroadcastOperator(odl.IdentityOperator(odl.rn(3)), 2),
     odl.ReductionOperator(odl.IdentityOperator(odl.rn(3)), 2),
     odl.DiagonalOperator(odl.IdentityOperator(odl.rn(3)), 2),
     ],
    fmt=' {name}={value.__class__.__name__}')


def test_pspace_op_init(base_op):
    """Test initialization with different base operators."""
    A = base_op

    op = odl.ProductSpaceOperator([[A]])
    assert op.domain == A.domain ** 1
    assert op.range == A.range ** 1

    op = odl.ProductSpaceOperator([[A, A]])
    assert op.domain == A.domain ** 2
    assert op.range == A.range ** 1

    op = odl.ProductSpaceOperator([[A],
                                   [A]])
    assert op.domain == A.domain ** 1
    assert op.range == A.range ** 2

    op = odl.ProductSpaceOperator([[A, 0],
                                   [0, A]])
    assert op.domain == A.domain ** 2
    assert op.range == A.range ** 2

    op = odl.ProductSpaceOperator([[A, None],
                                   [None, A]])
    assert op.domain == A.domain ** 2
    assert op.range == A.range ** 2


def test_pspace_op_derivative(base_op):
    """Test derivatives with different base operators."""
    A = base_op

    op = odl.ProductSpaceOperator([[A + 1]])
    true_deriv = odl.ProductSpaceOperator([[A]])
    deriv = op.derivative(op.domain.zero())
    assert deriv.domain == op.domain
    assert deriv.range == op.range
    x = op.domain.one()
    assert all_almost_equal(deriv(x), true_deriv(x))

    op = odl.ProductSpaceOperator([[A + 1, 2 * A - 1]])
    deriv = op.derivative(op.domain.zero())
    true_deriv = odl.ProductSpaceOperator([[A, 2 * A]])
    assert deriv.domain == op.domain
    assert deriv.range == op.range
    x = op.domain.one()
    assert all_almost_equal(deriv(x), true_deriv(x))


def test_pspace_op_adjoint(base_op):
    """Test adjoints with different base operators."""
    A = base_op

    op = odl.ProductSpaceOperator([[A]])
    true_adj = odl.ProductSpaceOperator([[A.adjoint]])
    adj = op.adjoint
    assert adj.domain == op.range
    assert adj.range == op.domain
    y = op.range.one()
    assert all_almost_equal(adj(y), true_adj(y))

    op = odl.ProductSpaceOperator([[2 * A, -A]])
    true_adj = odl.ProductSpaceOperator([[2 * A.adjoint],
                                         [-A.adjoint]])
    adj = op.adjoint
    assert adj.domain == op.range
    assert adj.range == op.domain
    y = op.range.one()
    assert all_almost_equal(adj(y), true_adj(y))


def test_pspace_op_weighted_init():

    r3 = odl.rn(3)
    ran = odl.ProductSpace(r3, 2, weighting=[1, 2])
    A = odl.IdentityOperator(r3)

    with pytest.raises(NotImplementedError):
        odl.ProductSpaceOperator([[A],
                                  [0]], range=ran)


def test_pspace_op_sum_call():
    r3 = odl.rn(3)
    A = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[A, A]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert all_almost_equal(op(z)[0], x + y)
    assert all_almost_equal(op(z, out=op.range.element())[0], x + y)


def test_pspace_op_project_call():
    r3 = odl.rn(3)
    A = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[A],
                                   [A]])

    x = r3.element([1, 2, 3])
    z = op.domain.element([x])

    assert all_almost_equal(op(z)[0], x)
    assert all_almost_equal(op(z, out=op.range.element())[0], x)
    assert all_almost_equal(op(z)[1], x)
    assert all_almost_equal(op(z, out=op.range.element())[1], x)


def test_pspace_op_diagonal_call():
    r3 = odl.rn(3)
    A = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[A, 0],
                                   [0, A]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])

    assert all_almost_equal(op(z), z)
    assert all_almost_equal(op(z, out=op.range.element()), z)


def test_pspace_op_swap_call():
    r3 = odl.rn(3)
    A = odl.IdentityOperator(r3)
    op = odl.ProductSpaceOperator([[0, A],
                                   [A, 0]])

    x = r3.element([1, 2, 3])
    y = r3.element([7, 8, 9])
    z = op.domain.element([x, y])
    true_result = op.domain.element([y, x])

    assert all_almost_equal(op(z), true_result)
    assert all_almost_equal(op(z, out=op.range.element()), true_result)


def test_comp_proj():
    r3 = odl.rn(3)
    r3xr3 = odl.ProductSpace(r3, 2)

    x = r3xr3.element([[1, 2, 3],
                       [4, 5, 6]])
    proj_0 = odl.ComponentProjection(r3xr3, 0)
    assert all_almost_equal(proj_0(x), x[0])
    assert all_almost_equal(proj_0(x, out=proj_0.range.element()), x[0])

    proj_1 = odl.ComponentProjection(r3xr3, 1)
    assert all_almost_equal(proj_1(x), x[1])
    assert all_almost_equal(proj_1(x, out=proj_1.range.element()), x[1])


def test_comp_proj_slice():
    r3 = odl.rn(3)
    r33 = odl.ProductSpace(r3, 3)

    x = r33.element([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    proj = odl.ComponentProjection(r33, slice(0, 2))
    assert all_almost_equal(proj(x), x[0:2])
    assert all_almost_equal(proj(x, out=proj.range.element()), x[0:2])


def test_comp_proj_indices():
    r3 = odl.rn(3)
    r33 = odl.ProductSpace(r3, 3)

    x = r33.element([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])

    proj = odl.ComponentProjection(r33, [0, 2])
    assert all_almost_equal(proj(x), x[[0, 2]])
    assert all_almost_equal(proj(x, out=proj.range.element()), x[[0, 2]])


def test_comp_proj_adjoint():
    r3 = odl.rn(3)
    r3xr3 = odl.ProductSpace(r3, 2)

    x = r3.element([1, 2, 3])

    result_0 = r3xr3.element([[1, 2, 3],
                              [0, 0, 0]])
    proj_0 = odl.ComponentProjection(r3xr3, 0)

    assert all_almost_equal(proj_0.adjoint(x), result_0)
    assert all_almost_equal(
        proj_0.adjoint(x, out=proj_0.domain.element()), result_0
    )

    result_1 = r3xr3.element([[0, 0, 0],
                              [1, 2, 3]])
    proj_1 = odl.ComponentProjection(r3xr3, 1)

    assert all_almost_equal(proj_1.adjoint(x), result_1)
    assert all_almost_equal(
        proj_1.adjoint(x, out=proj_1.domain.element()), result_1
    )


def test_comp_proj_adjoint_slice():
    r3 = odl.rn(3)
    r33 = odl.ProductSpace(r3, 3)

    x = r33[0:2].element([[1, 2, 3],
                          [4, 5, 6]])

    result = r33.element([[1, 2, 3],
                          [4, 5, 6],
                          [0, 0, 0]])
    proj = odl.ComponentProjection(r33, slice(0, 2))

    assert all_almost_equal(proj.adjoint(x), result)
    assert all_almost_equal(proj.adjoint(x, out=proj.domain.element()), result)


if __name__ == '__main__':
    odl.util.test_file(__file__)
