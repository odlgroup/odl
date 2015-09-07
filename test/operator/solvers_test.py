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
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()

# External module imports
import unittest
import numpy as np

# ODL imports
from odl.operator.operator import LinearOperator
import odl.operator.solvers as solvers
from odl.space.cartesian import Rn
from odl.utility.testutils import ODLTestCase


class MultiplyOp(LinearOperator):
    """Multiply with matrix
    """

    def __init__(self, matrix, domain=None, range=None):
        self._domain = (Rn(matrix.shape[1])
                        if domain is None else domain)
        self._range = (Rn(matrix.shape[0])
                       if range is None else range)
        self.matrix = matrix

    def _apply(self, rhs, out):
        out.data[:] = np.dot(self.matrix, rhs.data)

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range

    @property
    def adjoint(self):
        return MultiplyOp(self.matrix.T, self.range, self.domain)


class TestMatrixSolve(ODLTestCase):
    """ Tests solutions of the linear equation Ax = b with dense A
    """
    def test_landweber(self):
        n = 3

        # Np as validation
        A = np.random.rand(n, n)
        x = np.random.rand(n)
        # Landweber is slow and needs a decent initial guess
        b = np.dot(A, x) + 0.1 * np.random.rand(n)

        # Vector representation
        rn = Rn(n)
        xvec = rn.element(x)
        bvec = rn.element(b)

        # Make operator
        norm = np.linalg.norm(A, ord=2)
        Aop = MultiplyOp(A)

        # Solve using landweber
        solvers.landweber(Aop, xvec, bvec, niter=n*50, omega=1/norm**2)

        self.assertAllAlmostEquals(xvec, x, places=2)

    def test_conjugate_gradient(self):
        n = 3

        # Np as validation
        A = np.random.rand(n, n)
        x = np.random.rand(n)
        b = np.dot(A, x) + 0.1 * np.random.rand(n)

        # Vector representation
        rn = Rn(n)
        xvec = rn.element(x)
        bvec = rn.element(b)

        # Make operator
        Aop = MultiplyOp(A)

        # Solve using conjugate gradient
        solvers.conjugate_gradient(Aop, xvec, bvec, niter=n)

        self.assertAllAlmostEquals(xvec, x, places=2)

    def test_gauss_newton(self):
        n = 10

        # Np as validation
        A = np.random.rand(n, n)
        x = np.random.rand(n)
        b = np.dot(A, x) + 0.1 * np.random.rand(n)

        # Vector representation
        rn = Rn(n)
        xvec = rn.element(x)
        bvec = rn.element(b)

        # Make operator
        Aop = MultiplyOp(A)

        # Solve using conjugate gradient
        solvers.gauss_newton(Aop, xvec, bvec, niter=n)

        self.assertAllAlmostEquals(xvec, x, places=2)


if __name__ == '__main__':
    unittest.main(exit=False)
