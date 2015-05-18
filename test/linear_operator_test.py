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
import RL.operator.operator as op
from RL.space.euclidean import EuclideanSpace
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class MultiplyOp(op.LinearOperator):
    """Multiply with matrix
    """

    def __init__(self, matrix, domain=None, range=None):
        self._domain = (EuclideanSpace(matrix.shape[1])
                        if domain is None else domain)
        self._range = (EuclideanSpace(matrix.shape[0])
                       if range is None else range)
        self.matrix = matrix

    def _apply(self, rhs, out):
        np.dot(self.matrix, rhs.data, out=out.data)

    def _apply_adjoint(self, rhs, out):
        np.dot(self.matrix.T, rhs.data, out=out.data)

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range


class TestRN(RLTestCase):
    def test_MultiplyOp(self):
        # Verify that the multiply op does indeed work as expected

        A = np.random.rand(3, 3)
        x = np.random.rand(3)
        out = np.random.rand(3)

        Aop = MultiplyOp(A)
        xvec = Aop.domain.element(x)
        outvec = Aop.range.element()

        # Using apply
        Aop.apply(xvec, outvec)
        np.dot(A, x, out)
        self.assertAllAlmostEquals(out, outvec)

        # Using __call__
        self.assertAllAlmostEquals(Aop(xvec), np.dot(A, x))

    def test_MultiplyOp_nonsquare(self):
        # Verify that the multiply op does indeed work as expected
        A = np.random.rand(4, 3)
        x = np.random.rand(3)
        out = np.random.rand(4)

        Aop = MultiplyOp(A)
        xvec = Aop.domain.element(x)
        outvec = Aop.range.element()

        # Using apply
        Aop.apply(xvec, outvec)
        np.dot(A, x, out)
        self.assertAllAlmostEquals(out, outvec)

        # Using __call__
        self.assertAllAlmostEquals(Aop(xvec), np.dot(A, x))

    def test_adjoint(self):
        A = np.random.rand(4, 3)
        x = np.random.rand(4)
        out = np.random.rand(3)

        Aop = MultiplyOp(A)
        xvec = Aop.range.element(x)
        outvec = Aop.domain.element()

        # Using apply_adjoint
        Aop.apply_adjoint(xvec, outvec)
        np.dot(A.T, x, out)
        self.assertAllAlmostEquals(out, outvec)

        # By creating an OperatorAdjoint object
        self.assertAllAlmostEquals(op.OperatorAdjoint(Aop)(xvec),
                                   np.dot(A.T, x))

        # Using T method and __call__
        self.assertAllAlmostEquals(Aop.T(xvec), np.dot(A.T, x))

    def test_addition(self):
        A = np.random.rand(4, 3)
        B = np.random.rand(4, 3)
        x = np.random.rand(3)
        y = np.random.rand(4)

        Aop = MultiplyOp(A)
        Bop = MultiplyOp(B)
        xvec = Aop.domain.element(x)
        yvec = Aop.range.element(y)

        # Explicit instantiation
        C = op.LinearOperatorSum(Aop, Bop)

        self.assertAllAlmostEquals(C(xvec), np.dot(A, x) + np.dot(B, x))
        self.assertAllAlmostEquals(C.T(yvec), np.dot(A.T, y) + np.dot(B.T, y))

        # Using operator overloading
        self.assertAllAlmostEquals((Aop + Bop)(xvec),
                                   np.dot(A, x) + np.dot(B, x))
        self.assertAllAlmostEquals((Aop + Bop).T(yvec),
                                   np.dot(A.T, y) + np.dot(B.T, y))

    def test_scale(self):
        A = np.random.rand(4, 3)
        x = np.random.rand(3)
        y = np.random.rand(4)

        Aop = MultiplyOp(A)
        xvec = Aop.domain.element(x)
        yvec = Aop.range.element(y)

        # Test a range of scalars (scalar multiplication could implement
        # optimizations for (-1, 0, 1).
        scalars = [-1.432, -1, 0, 1, 3.14]
        for scale in scalars:
            C = op.LinearOperatorScalarMultiplication(Aop, scale)

            self.assertAllAlmostEquals(C(xvec), scale * np.dot(A, x))
            self.assertAllAlmostEquals(C.T(yvec), scale * np.dot(A.T, y))

            # Using operator overloading
            self.assertAllAlmostEquals((scale * Aop)(xvec),
                                       scale * np.dot(A, x))
            self.assertAllAlmostEquals((Aop * scale)(xvec),
                                       np.dot(A, scale * x))
            self.assertAllAlmostEquals((scale * Aop).T(yvec),
                                       scale * np.dot(A.T, y))
            self.assertAllAlmostEquals((Aop * scale).T(yvec),
                                       np.dot(A.T, scale * y))

    def test_composition(self):
        A = np.random.rand(5, 4)
        B = np.random.rand(4, 3)
        x = np.random.rand(3)
        y = np.random.rand(5)

        Aop = MultiplyOp(A)
        Bop = MultiplyOp(B)
        xvec = Bop.domain.element(x)
        yvec = Aop.range.element(y)

        C = op.LinearOperatorComposition(Aop, Bop)

        self.assertAllAlmostEquals(C(xvec), np.dot(A, np.dot(B, x)))
        self.assertAllAlmostEquals(C.T(yvec), np.dot(B.T, np.dot(A.T, y)))

    def test_type_errors(self):
        r3 = EuclideanSpace(3)
        r4 = EuclideanSpace(4)

        Aop = MultiplyOp(np.random.rand(3, 3))
        r3Vec1 = r3.zero()
        r3Vec2 = r3.zero()
        r4Vec1 = r4.zero()
        r4Vec2 = r4.zero()

        # Verify that correct usage works
        Aop.apply(r3Vec1, r3Vec2)
        Aop.apply_adjoint(r3Vec1, r3Vec2)

        # Test that erroneous usage raises TypeError
        with self.assertRaises(TypeError):
            Aop(r4Vec1)

        with self.assertRaises(TypeError):
            Aop.T(r4Vec1)

        with self.assertRaises(TypeError):
            Aop.apply(r3Vec1, r4Vec1)

        with self.assertRaises(TypeError):
            Aop.apply_adjoint(r3Vec1, r4Vec1)

        with self.assertRaises(TypeError):
            Aop.apply(r4Vec1, r3Vec1)

        with self.assertRaises(TypeError):
            Aop.apply_adjoint(r4Vec1, r3Vec1)

        with self.assertRaises(TypeError):
            Aop.apply(r4Vec1, r4Vec2)

        with self.assertRaises(TypeError):
            Aop.apply_adjoint(r4Vec1, r4Vec2)

        # Check test against aliased values
        with self.assertRaises(ValueError):
            Aop.apply(r3Vec1, r3Vec1)

        with self.assertRaises(ValueError):
            Aop.apply_adjoint(r3Vec1, r3Vec1)


if __name__ == '__main__':
    unittest.main(exit=False)
