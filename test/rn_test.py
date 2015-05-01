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
# import RL.operator.operator as op
# import RL.space.space as space
from RL.space.euclidean import RN
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class RNTest(RLTestCase):
    def makeVectors(self, rn):
        # Generate numpy vectors
        y = np.random.rand(rn.dimension)
        x = np.random.rand(rn.dimension)
        z = np.random.rand(rn.dimension)

        # Make rn vectors
        yVec = rn.makeVector(y)
        xVec = rn.makeVector(x)
        zVec = rn.makeVector(z)
        return x, y, z, xVec, yVec, zVec

    def doLincombTest(self, a, b, n=10):
        # Validates lincomb against the result on host with randomized
        # data and given a,b
        rn = RN(n)

        # Unaliased arguments
        x, y, z, xVec, yVec, zVec = self.makeVectors(rn)

        z[:] = a*x + b*y
        rn.linComb(zVec, a, xVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # First argument aliased with output
        x, y, z, xVec, yVec, zVec = self.makeVectors(rn)

        z[:] = a*z + b*y
        rn.linComb(zVec, a, zVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # Second argument aliased with output
        x, y, z, xVec, yVec, zVec = self.makeVectors(rn)

        z[:] = a*x + b*z
        rn.linComb(zVec, a, xVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # Both arguments aliased with each other
        x, y, z, xVec, yVec, zVec = self.makeVectors(rn)

        z[:] = a*x + b*x
        rn.linComb(zVec, a, xVec, b, xVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # All aliased
        x, y, z, xVec, yVec, zVec = self.makeVectors(rn)
        z[:] = a*z + b*z
        rn.linComb(zVec, a, zVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

    def testLinComb(self):
        scalar_values = [0, 1, -1, 3.41]
        for a in scalar_values:
            for b in scalar_values:
                self.doLincombTest(a, b)


class OperatorOverloadTest(RLTestCase):
    def doUnaryOperatorTest(self, function, n=10):
        """ Verifies that the statement y=function(x) gives equivalent
        results to numpy.
        """
        x = np.random.rand(n)

        rn = RN(n)
        xVec = rn.makeVector(x)

        y = function(x)
        yVec = function(xVec)

        self.assertAllAlmostEquals(xVec, x)
        self.assertAllAlmostEquals(yVec, y)

    def doBinaryOperatorTest(self, function, n=10):
        """ Verifies that the statement z=function(x,y) gives equivalent
        results to numpy.
        """
        y = np.random.rand(n)
        x = np.random.rand(n)

        rn = RN(n)
        yVec = rn.makeVector(y)
        xVec = rn.makeVector(x)

        z = function(x, y)
        zVec = function(xVec, yVec)

        self.assertAllAlmostEquals(xVec, x)
        self.assertAllAlmostEquals(yVec, y)
        self.assertAllAlmostEquals(zVec, z)

    def testOperators(self):
        """ Test of all operator overloads against the corresponding
        numpy implementation
        """
        # Unary operators
        self.doUnaryOperatorTest(lambda x: +x)
        self.doUnaryOperatorTest(lambda x: -x)

        # Scalar multiplication
        for scalar in [-31.2, -1, 0, 1, 2.13]:
            def incMul(x): x *= scalar
            self.doUnaryOperatorTest(incMul)
            self.doUnaryOperatorTest(lambda x: x*scalar)

        # Scalar division
        for scalar in [-31.2, -1, 1, 2.13]:
            def incDiv(x): x /= scalar
            self.doUnaryOperatorTest(incDiv)
            self.doUnaryOperatorTest(lambda x: x/scalar)

        # Incremental operations
        def incAdd(x, y):
            x += y

        def incSub(x, y):
            x -= y

        self.doBinaryOperatorTest(incAdd)
        self.doBinaryOperatorTest(incSub)

        # Incremental operators with aliased inputs
        def incAddAliased(x):
            x += x

        def incSubAliased(x):
            x -= x
        self.doUnaryOperatorTest(incAddAliased)
        self.doUnaryOperatorTest(incSubAliased)

        # Binary operators
        self.doBinaryOperatorTest(lambda x, y: x + y)
        self.doBinaryOperatorTest(lambda x, y: x - y)

        # Binary with aliased inputs
        self.doUnaryOperatorTest(lambda x: x + x)
        self.doUnaryOperatorTest(lambda x: x - x)


if __name__ == '__main__':
    unittest.main(exit=False)
