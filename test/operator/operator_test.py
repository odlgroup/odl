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
try:
    from builtins import str
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str

# External module imports
import unittest
import numpy as np

# RL imports
import RL.operator.operator as op
from RL.space.euclidean import EuclideanSpace
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class MultiplyAndSquareOp(op.Operator):
    """ Example of a nonlinear operator, Calculates (A*x)**2
    """

    def __init__(self, matrix, domain=None, range=None):
        self.domain = (EuclideanSpace(matrix.shape[1])
                        if domain is None else domain)
        self.range = (EuclideanSpace(matrix.shape[0])
                       if range is None else range)
        self.matrix = matrix

    def _apply(self, rhs, out):
        np.dot(self.matrix, rhs.data, out=out.data)
        out.data[:] **= 2

    def __str__(self):
        return "MaS: " + str(self.matrix) + "**2"


def mult_sq_np(A, x):
    # The same as MultiplyAndSquareOp but only using numpy
    return np.dot(A, x)**2


class TestRN(RLTestCase):
    def test_mult_sq_op(self):
        # Verify that the operator does indeed work as expected
        A = np.random.rand(4, 3)
        x = np.random.rand(3)
        Aop = MultiplyAndSquareOp(A)
        xvec = Aop.domain.element(x)

        self.assertAllAlmostEquals(Aop(xvec), mult_sq_np(A, x))

    def test_addition(self):
        # Test operator addition
        A = np.random.rand(4, 3)
        B = np.random.rand(4, 3)
        x = np.random.rand(3)

        Aop = MultiplyAndSquareOp(A)
        Bop = MultiplyAndSquareOp(B)
        xvec = Aop.domain.element(x)

        # Explicit instantiation
        C = op.OperatorSum(Aop, Bop)
        self.assertAllAlmostEquals(C(xvec),
                                   mult_sq_np(A, x) + mult_sq_np(B, x))

        # Using operator overloading
        self.assertAllAlmostEquals((Aop + Bop)(xvec),
                                   mult_sq_np(A, x) + mult_sq_np(B, x))

        # Verify that unmatched operators domains fail
        C = np.random.rand(4, 4)
        Cop = MultiplyAndSquareOp(C)

        with self.assertRaises(TypeError):
            C = op.OperatorSum(Aop, Cop)

    def test_scale(self):
        A = np.random.rand(4, 3)
        x = np.random.rand(3)

        Aop = MultiplyAndSquareOp(A)
        xvec = Aop.domain.element(x)

        # Test a range of scalars (scalar multiplication could implement
        # optimizations for (-1, 0, 1)).
        scalars = [-1.432, -1, 0, 1, 3.14]
        for scale in scalars:
            lscaled = op.OperatorLeftScalarMultiplication(Aop, scale)
            rscaled = op.OperatorRightScalarMultiplication(Aop, scale)

            self.assertAllAlmostEquals(lscaled(xvec),
                                       scale * mult_sq_np(A, x))
            self.assertAllAlmostEquals(rscaled(xvec),
                                       mult_sq_np(A, scale*x))

            # Using operator overloading
            self.assertAllAlmostEquals((scale * Aop)(xvec),
                                       scale * mult_sq_np(A, x))
            self.assertAllAlmostEquals((Aop * scale)(xvec),
                                       mult_sq_np(A, scale*x))

        # Fail when scaling by wrong scalar type (A complex number)
        nonscalars = [1j, [1, 2], Aop]
        for nonscalar in nonscalars:
            with self.assertRaises(TypeError):
                C = op.OperatorLeftScalarMultiplication(Aop, nonscalar)

            with self.assertRaises(TypeError):
                C = op.OperatorRightScalarMultiplication(Aop, nonscalar)

            with self.assertRaises(TypeError):
                C = Aop * nonscalar

            with self.assertRaises(TypeError):
                C = nonscalar * Aop

    def test_composition(self):
        A = np.random.rand(5, 4)
        B = np.random.rand(4, 3)
        x = np.random.rand(3)

        Aop = MultiplyAndSquareOp(A)
        Bop = MultiplyAndSquareOp(B)
        xvec = Bop.domain.element(x)

        C = op.OperatorComposition(Aop, Bop)

        self.assertAllAlmostEquals(C(xvec), mult_sq_np(A, mult_sq_np(B, x)))

        # Verify that incorrect order fails
        with self.assertRaises(TypeError):
            C = op.OperatorComposition(Bop, Aop)


if __name__ == '__main__':
    unittest.main(exit=False)
