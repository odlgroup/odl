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
from builtins import super

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl.operator.oputils import matrix_representation, power_method_opnorm
from odl.space.pspace import ProductSpace
from odl.operator.pspace_ops import ProductSpaceOperator

from odl.space.npy_ntuples import MatVecOperator
from odl.util.testutils import almost_equal


def test_matrix_representation():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    A = np.random.rand(n, n)

    Aop = MatVecOperator(A)

    the_matrix = matrix_representation(Aop)

    assert almost_equal(np.sum(np.abs(A - the_matrix)), 1e-6)


def test_matrix_representation_product_to_lin_space():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = MatVecOperator(A)

    m = 2
    rm = odl.rn(m)
    B = np.random.rand(n, m)
    Bop = MatVecOperator(B)

    dom = ProductSpace(rn, rm)
    ran = ProductSpace(rn, 1)

    AB_matrix = np.hstack([A, B])
    ABop = ProductSpaceOperator([Aop, Bop], dom, ran)

    the_matrix = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - the_matrix)), 1e-6)


def test_matrix_representation_lin_space_to_product():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = MatVecOperator(A)

    m = 2
    rm = odl.rn(m)
    B = np.random.rand(m, n)
    Bop = MatVecOperator(B)

    dom = ProductSpace(rn, 1)
    ran = ProductSpace(rn, rm)

    AB_matrix = np.vstack([A, B])
    ABop = ProductSpaceOperator([[Aop], [Bop]], dom, ran)

    the_matrix = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - the_matrix)), 1e-6)


def test_matrix_representation_product_to_product():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = MatVecOperator(A)

    m = 2
    rm = odl.rn(m)
    B = np.random.rand(m, m)
    Bop = MatVecOperator(B)

    ran_and_dom = ProductSpace(rn, rm)

    AB_matrix = np.vstack([np.hstack([A, np.zeros((n, m))]),
                          np.hstack([np.zeros((m, n)), B])])
    ABop = ProductSpaceOperator([[Aop, 0],
                                 [0, Bop]],
                                ran_and_dom, ran_and_dom)
    the_matrix = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - the_matrix)), 1e-6)


def test_matrix_representation_product_to_product_two():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.rn(n)
    A = np.random.rand(n, n)
    Aop = MatVecOperator(A)

    B = np.random.rand(n, n)
    Bop = MatVecOperator(B)

    ran_and_dom = ProductSpace(rn, 2)

    AB_matrix = np.vstack([np.hstack([A, np.zeros((n, n))]),
                          np.hstack([np.zeros((n, n)), B])])
    ABop = ProductSpaceOperator([[Aop, 0],
                                 [0, Bop]],
                                ran_and_dom, ran_and_dom)
    the_matrix = matrix_representation(ABop)

    assert almost_equal(np.sum(np.abs(AB_matrix - the_matrix)), 1e-6)


def test_matrix_representation_not_linear_op():
    # Verify that the matrix representation function gives correct error
    class small_nonlin_op(odl.Operator):
        """Small nonlinear test operator"""
        def __init__(self):
            super().__init__(domain=odl.rn(3), range=odl.rn(4), linear=False)

        def _call(self, x):
            return x

    nonlin_op = small_nonlin_op()
    with pytest.raises(ValueError):
        matrix_representation(nonlin_op)


def test_matrix_representation_wrong_domain():
    # Verify that the matrix representation function gives correct error
    class small_op(odl.Operator):
        """Small nonlinear test operator"""
        def __init__(self):
            super().__init__(domain=ProductSpace(odl.rn(3),
                                                 ProductSpace(odl.rn(3),
                                                              odl.rn(3))),
                             range=odl.rn(4), linear=True)

        def _call(self, x, out):
            return odl.rn(np.random.rand(4))

    nonlin_op = small_op()
    with pytest.raises(TypeError):
        matrix_representation(nonlin_op)


def test_matrix_representation_wrong_range():
    # Verify that the matrix representation function gives correct error
    class small_op(odl.Operator):
        """Small nonlinear test operator"""
        def __init__(self):
            super().__init__(domain=odl.rn(3),
                             range=ProductSpace(odl.rn(3),
                                                ProductSpace(odl.rn(3),
                                                             odl.rn(3))),
                             linear=True)

        def _call(self, x, out):
            return odl.rn(np.random.rand(4))

    nonlin_op = small_op()
    with pytest.raises(TypeError):
        matrix_representation(nonlin_op)


def test_power_method_opnorm_symm():
    # Test the power method on a matrix operator

    # Test matrix with eigenvalues 1 and -2
    # Rather nasty case since the eigenvectors are almost parallel
    mat = np.array([[10, -18],
                    [6, -11]], dtype=float)

    op = odl.MatVecOperator(mat)
    true_opnorm = 2
    opnorm_est = power_method_opnorm(op)
    assert almost_equal(opnorm_est, true_opnorm, places=2)

    # Start at a different point
    xstart = odl.rn(2).element([0.8, 0.5])
    opnorm_est = power_method_opnorm(op, xstart=xstart)
    assert almost_equal(opnorm_est, true_opnorm, places=2)


def test_power_method_opnorm_nonsymm():
    # Test the power method on a matrix operator

    # Singular values 5.5 and 6
    mat = np.array([[-1.52441557, 5.04276365],
                    [1.90246927, 2.54424763],
                    [5.32935411, 0.04573162]])

    op = odl.MatVecOperator(mat)
    true_opnorm = 6

    # Start vector (1, 1) is close to the wrong eigenvector
    opnorm_est = power_method_opnorm(op, maxiter=50)
    assert almost_equal(opnorm_est, true_opnorm, places=2)

    # Start close to the correct eigenvector, converges very fast
    xstart = odl.rn(2).element([-0.8, 0.5])
    opnorm_est = power_method_opnorm(op, maxiter=6, xstart=xstart)
    assert almost_equal(opnorm_est, true_opnorm, places=2)


def test_power_method_opnorm_exceptions():
    # Test the exceptions

    space = odl.rn(2)
    op = odl.IdentityOperator(space)

    with pytest.raises(ValueError):
        # Too small number of iterates
        power_method_opnorm(op, maxiter=0)

    with pytest.raises(ValueError):
        # Negative number of iterates
        power_method_opnorm(op, maxiter=-5)

    with pytest.raises(ValueError):
        # Input vector is zero
        power_method_opnorm(op, maxiter=2, xstart=space.zero())

    with pytest.raises(ValueError):
        # Input vector in the nullspace
        op = odl.MatVecOperator([[0., 1.],
                                 [0., 0.]])

        power_method_opnorm(op, maxiter=2, xstart=op.domain.one())

    with pytest.raises(ValueError):
        # Uneven number of iterates for non square operator
        op = odl.MatVecOperator([[1., 2., 3.],
                                 [4., 5., 6.]])

        power_method_opnorm(op, maxiter=1, xstart=op.domain.one())

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
