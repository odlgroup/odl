# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test for the alternating dual updates solver."""

from __future__ import division
import odl
from odl.solvers.nonsmooth import adupdates
from odl.solvers.nonsmooth.alternating_dual_updates import adupdates_simple

from odl.util.testutils import all_almost_equal


# Places for the accepted error when comparing results
HIGH_ACCURACY = 8
LOW_ACCURACY = 4


def test_adupdates():
    """Test if the adupdates solver handles the following problem correctly:

    ( 1  1/2 1/3 1/4) (x_1)   (a)
    (1/2 1/3 1/4 1/5) (x_2)   (b)
    (1/3 1/4 1/5 1/6) (x_3) = (c)
    (1/4 1/5 1/6 1/7) (x_4)   (d).

    The matrix is the ill-conditined Hilbert matrix, the inverse of which
    can be given in closed form. If we set

    (a)   (25/12)                    (x_1) = (1)
    (b)   (77/60)                    (x_2) = (1)
    (c) = (19/20)         then       (x_3) = (1)
    (d)   (319/420),                 (x_4) = (1).

    We solve the problem

    min ||Ax - b||^2 + TV(x) s.t. x >= 0

    for the matrix A, the r.h.s. b as above and the total variation TV, which
    is given as the (non-cyclic) sum of the distances of consecutive entries
    of the solution. The solution of this problem is clearly x = (1, 1, 1, 1),
    since it satisfies the additional constraint and minimizes both terms of
    the objective function.
    """

    mat1 = [[1, 1 / 2, 1 / 3, 1 / 4],
            [1 / 2, 1 / 3, 1 / 4, 1 / 5]]
    mat2 = [[1 / 3, 1 / 4, 1 / 5, 1 / 6],
            [1 / 4, 1 / 5, 1 / 6, 1 / 7]]

    # Create the linear operators
    mat1op = odl.MatrixOperator(mat1)
    mat2op = odl.MatrixOperator(mat2)
    domain = mat1op.domain
    tv1 = odl.MatrixOperator([[1.0, -1.0, 0.0, 0.0]])
    tv2 = odl.MatrixOperator([[0.0, 0.0, 1.0, -1.0]])
    tv3 = odl.MatrixOperator([[0.0, 1.0, -1.0, 0.0]])
    nneg = odl.IdentityOperator(domain)
    ops = [mat1op, mat2op, odl.BroadcastOperator(tv1, tv2), tv3, nneg]

    # Create inner stepsizes for linear operators
    mat1s = 1 / mat1op(mat1op.adjoint(mat1op.range.one()))
    mat2s = 1 / mat2op(mat2op.adjoint(mat2op.range.one()))
    tv1s = [0.5, 0.5]
    tv2s = 0.5
    nnegs = nneg.range.element([1.0, 1.0, 1.0, 1.0])
    inner_stepsizes = [mat1s, mat2s, tv1s, tv2s, nnegs]

    expected_solution = domain.element([1, 1, 1, 1])
    # Create right-hand-sides of the equation
    rhs1 = mat1op(expected_solution)
    rhs2 = mat2op(expected_solution)

    # Create the functionals
    fid1 = odl.solvers.L2NormSquared(mat1op.range).translated(rhs1)
    fid2 = odl.solvers.L2NormSquared(mat2op.range).translated(rhs2)
    reg1 = odl.solvers.L1Norm(tv1.range)
    reg2 = odl.solvers.L1Norm(tv2.range)
    reg3 = odl.solvers.L1Norm(tv3.range)
    ind = odl.solvers.IndicatorNonnegativity(nneg.range)
    funcs = [fid1, fid2, odl.solvers.SeparableSum(reg1, reg2), reg3, ind]

    # Start from zero
    x = tv1.domain.zero()
    x_simple = tv1.domain.zero()

    stepsize = 1.0
    niter = 10

    adupdates(x, funcs, ops, stepsize, inner_stepsizes, niter)
    adupdates_simple(x_simple, funcs, ops, stepsize,
                     inner_stepsizes, niter)
    assert all_almost_equal(x, x_simple)
    assert domain.dist(x, expected_solution) < 3e-2


if __name__ == '__main__':
    odl.util.test_file(__file__)
