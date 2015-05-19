# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import
from future import standard_library

# External module imports
import unittest
import numpy as np

# RL imports
import RL.operator.operator as OP
import RL.operator.solvers as solvers
from RL.space.space import *
from RL.space.euclidean import *
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class MultiplyOp(OP.LinearOperator):
    """Multiply with matrix
    """

    def __init__(self, matrix, domain=None, range=None):
        self.domain = (EuclideanSpace(matrix.shape[1])
                        if domain is None else domain)
        self.range = (EuclideanSpace(matrix.shape[0])
                       if range is None else range)
        self.matrix = matrix

    def _apply(self, rhs, out):
        out.values[:] = np.dot(self.matrix, rhs.values)

    @property
    def adjoint(self):
        return MultiplyOp(self.matrix.T, self.range, self.domain)


class TestMatrixSolve(RLTestCase):
    """ Tests solutions of the linear equation Ax = b with dense A
    """
    def testLandweber(self):
        n = 3

        # Np as validation
        A = np.random.rand(n, n)
        x = np.random.rand(n)
        # Landweber is slow and needs a decent initial guess
        b = np.dot(A, x) + 0.1 * np.random.rand(n)

        # Vector representation
        rn = EuclideanSpace(n)
        xVec = rn.element(x)
        bVec = rn.element(b)

        # Make operator
        norm = np.linalg.norm(A, ord=2)
        Aop = MultiplyOp(A)

        # Solve using landweber
        solvers.landweber(Aop, xVec, bVec, iterations=n*50, omega=1/norm**2)

        self.assertAllAlmostEquals(xVec, x, places=2)

    def testCGN(self):
        n = 3

        # Np as validation
        A = np.random.rand(n, n)
        x = np.random.rand(n)
        b = np.dot(A, x) + 0.1 * np.random.rand(n)

        # Vector representation
        rn = EuclideanSpace(n)
        xVec = rn.element(x)
        bVec = rn.element(b)

        # Make operator
        Aop = MultiplyOp(A)

        # Solve using conjugate gradient
        solvers.conjugateGradient(Aop, xVec, bVec, iterations=n)

        self.assertAllAlmostEquals(xVec, x, places=2)

    def testGaussNewton(self):
        n = 100

        # Np as validation
        A = np.random.rand(n, n)
        x = np.random.rand(n)
        b = np.dot(A, x) + 0.1 * np.random.rand(n)

        # Vector representation
        rn = EuclideanSpace(n)
        xVec = rn.element(x)
        bVec = rn.element(b)

        # Make operator
        Aop = MultiplyOp(A)

        # Solve using conjugate gradient
        solvers.gaussNewton(Aop, xVec, bVec, iterations=n)

        self.assertAllAlmostEquals(xVec, x, places=2)


if __name__ == '__main__':
    unittest.main(exit=False)
