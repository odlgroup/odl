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
from math import sqrt

# RL imports
# import RL.operator.operator as op
# import RL.space.space as space
from RL.space.cartesian import Rn, MetricRn, NormedRn, EuclideanRn, cartesian
from RL.utility.testutils import RLTestCase, skip_all_tests

try:
    from RL.space.cuda import *
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

standard_library.install_aliases()


class RnTest(RLTestCase):
    @staticmethod
    def _vectors(rn):
        # Generate numpy vectors
        y = np.random.rand(rn.dim)
        x = np.random.rand(rn.dim)
        z = np.random.rand(rn.dim)

        # Make rn vectors
        yVec = rn.element(y)
        xVec = rn.element(x)
        zVec = rn.element(z)
        return x, y, z, xVec, yVec, zVec

    def _test_lincomb(self, a, b, n=10):
        # Validates lincomb against the result on host with randomized
        # data and given a,b
        rn = Rn(n)

        # Unaliased arguments
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*x + b*y
        rn.lincomb(zVec, a, xVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # First argument aliased with output
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*z + b*y
        rn.lincomb(zVec, a, zVec, b, yVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # Second argument aliased with output
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*x + b*z
        rn.lincomb(zVec, a, xVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # Both arguments aliased with each other
        x, y, z, xVec, yVec, zVec = self._vectors(rn)

        z[:] = a*x + b*x
        rn.lincomb(zVec, a, xVec, b, xVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

        # All aliased
        x, y, z, xVec, yVec, zVec = self._vectors(rn)
        z[:] = a*z + b*z
        rn.lincomb(zVec, a, zVec, b, zVec)
        self.assertAllAlmostEquals([xVec, yVec, zVec], [x, y, z])

    def test_lincomb(self):
        scalar_values = [0, 1, -1, 3.41]
        for a in scalar_values:
            for b in scalar_values:
                self._test_lincomb(a, b)


class OperatorOverloadTest(RLTestCase):
    def _test_unary_operator(self, function, n=10):
        """ Verifies that the statement y=function(x) gives equivalent
        results to Numpy.
        """
        rn = Rn(n)

        x_arr = np.random.rand(n)
        y_arr = function(x_arr)

        x = rn.element(x_arr)
        y = function(x)

        self.assertAllAlmostEquals(x, x_arr)
        self.assertAllAlmostEquals(y, y_arr)

    def _test_binary_operator(self, function, n=10):
        """ Verifies that the statement z=function(x,y) gives equivalent
        results to Numpy.
        """
        rn = Rn(n)

        x_arr = np.random.rand(n)
        y_arr = np.random.rand(n)
        z_arr = function(x_arr, y_arr)

        x = rn.element(x_arr)
        y = rn.element(y_arr)
        z = function(x, y)

        self.assertAllAlmostEquals(x, x_arr)
        self.assertAllAlmostEquals(y, y_arr)
        self.assertAllAlmostEquals(z, z_arr)

    def test_operators(self):
        """ Test of all operator overloads against the corresponding
        Numpy implementation
        """
        # Unary operators
        self._test_unary_operator(lambda x: +x)
        self._test_unary_operator(lambda x: -x)

        # Scalar multiplication
        for scalar in [-31.2, -1, 0, 1, 2.13]:
            def imul(x):
                x *= scalar
            self._test_unary_operator(imul)
            self._test_unary_operator(lambda x: x*scalar)

        # Scalar division
        for scalar in [-31.2, -1, 1, 2.13]:
            def idiv(x):
                x /= scalar
            self._test_unary_operator(idiv)
            self._test_unary_operator(lambda x: x/scalar)

        # Incremental operations
        def iadd(x, y):
            x += y

        def isub(x, y):
            x -= y

        self._test_binary_operator(iadd)
        self._test_binary_operator(isub)

        # Incremental operators with aliased inputs
        def iadd_aliased(x):
            x += x

        def isub_aliased(x):
            x -= x
        self._test_unary_operator(iadd_aliased)
        self._test_unary_operator(isub_aliased)

        # Binary operators
        self._test_binary_operator(lambda x, y: x + y)
        self._test_binary_operator(lambda x, y: x - y)

        # Binary with aliased inputs
        self._test_unary_operator(lambda x: x + x)
        self._test_unary_operator(lambda x: x - x)


class MethodTest(RLTestCase):
    def test_norm(self):
        r3 = EuclideanRn(3)
        xd = r3.element([1, 2, 3])

        correct_norm = sqrt(1**2 + 2**2 + 3**2)

        # Space function
        self.assertAlmostEquals(r3.norm(xd), correct_norm)

        # Member function
        self.assertAlmostEquals(xd.norm(), correct_norm)

    def test_inner(self):
        r3 = EuclideanRn(3)
        xd = r3.element([1, 2, 3])
        yd = r3.element([5, -3, 9])

        correct_inner = 1*5 + 2*(-3) + 3*9

        # Space function
        self.assertAlmostEquals(r3.inner(xd, yd), correct_inner)

        # Member function
        self.assertAlmostEquals(xd.inner(yd), correct_inner)


class CpuFactoryTest(RLTestCase):
    def test_plain(self):
        r3 = Rn(3)
        r3_fac = cartesian(3, dist=False)

        # Space type
        self.assertEqual(r3, r3_fac)

        # Elements
        arr = np.random.rand(r3.dim)
        x = r3.element(arr)
        x_fac = r3_fac.element(arr)
        self.assertAllEquals(x, x_fac)

    @staticmethod
    def _dist(x, y):
        return np.sum(np.abs(x - y))

    def test_metric(self):
        r3 = MetricRn(3, dist=self._dist)
        r3_fac = cartesian(3, dist=self._dist)

        # Space type
        self.assertEqual(r3, r3_fac)

        # Distance function
        arr_x = np.random.rand(r3.dim)
        x = r3.element(arr_x)
        x_fac = r3_fac.element(arr_x)
        arr_y = np.random.rand(r3.dim)
        y = r3.element(arr_y)
        y_fac = r3_fac.element(arr_y)

        self.assertEqual(x.dist(y), x_fac.dist(y_fac))
        self.assertEqual(x.dist(y_fac), x_fac.dist(y))

    @staticmethod
    def _norm(x):
        return np.sum(np.abs(x))

    def test_norm(self):
        r3 = NormedRn(3, norm=self._norm)
        r3_fac = cartesian(3, norm=self._norm)

        # Space type
        self.assertEqual(r3, r3_fac)

        # Norm function
        arr = np.random.rand(r3.dim)
        x = r3.element(arr)
        x_fac = r3_fac.element(arr)

        self.assertEqual(x.norm(), x_fac.norm())

    def test_p_norm(self):
        r3 = NormedRn(3, p=1.5)
        r3_fac = cartesian(3, norm_p=1.5)

        # Space type
        self.assertEqual(r3, r3_fac)

        # Norm function
        arr = np.random.rand(r3.dim)
        x = r3.element(arr)
        x_fac = r3_fac.element(arr)

        self.assertEqual(x.norm(), x_fac.norm())

    @staticmethod
    def _inner(x, y):
        w = np.arange(1, len(x)+1, dtype=np.float64) / float(len(x))
        return np.sum(np.dot(x, w*y))

    def test_inner(self):
        r3 = EuclideanRn(3, inner=self._inner)
        r3_fac = cartesian(3, inner=self._inner)

        # Space type
        self.assertEqual(r3, r3_fac)

        # Inner product function
        arr_x = np.random.rand(r3.dim)
        x = r3.element(arr_x)
        x_fac = r3_fac.element(arr_x)
        arr_y = np.random.rand(r3.dim)
        y = r3.element(arr_y)
        y_fac = r3_fac.element(arr_y)

        self.assertEqual(x.inner(y), x_fac.inner(y_fac))
        self.assertEqual(x.inner(y_fac), x_fac.inner(y))

    def test_weights(self):
        w = np.arange(1, 4, dtype=np.float64) / 3.0

        # Normed space
        r3n = NormedRn(3, p=1.5, weights=w)
        r3n_fac = cartesian(3, norm_p=1.5, weights=w)

        # Space type
        self.assertEqual(r3n, r3n_fac)

        # Norm function
        arr = np.random.rand(r3n.dim)
        x = r3n.element(arr)
        x_fac = r3n_fac.element(arr)

        self.assertEqual(x.norm(), x_fac.norm())

        # Inner product space
        r3i = EuclideanRn(3, weights=w)
        r3i_fac = cartesian(3, weights=w)

        # Space type
        self.assertEqual(r3i, r3i_fac)

        # Inner product function
        arr_x = np.random.rand(r3i.dim)
        x = r3i.element(arr_x)
        x_fac = r3i_fac.element(arr_x)
        arr_y = np.random.rand(r3i.dim)
        y = r3i.element(arr_y)
        y_fac = r3i_fac.element(arr_y)

        self.assertEqual(x.inner(y), x_fac.inner(y_fac))
        self.assertEqual(x.inner(y_fac), x_fac.inner(y))


if not CUDA_AVAILABLE:
    RLTestCase = skip_all_tests("Missing RLcpp")


class GpuFactoryTest(RLTestCase):
    def test_inner(self):
        r3 = CudaRn(3)
        r3_fac = cartesian(3, impl='cuda')

        # Space type
        self.assertEqual(r3, r3_fac)

        # Inner product function
        arr_x = np.random.rand(r3.dim)
        x = r3.element(arr_x)
        x_fac = r3_fac.element(arr_x)
        arr_y = np.random.rand(r3.dim)
        y = r3.element(arr_y)
        y_fac = r3_fac.element(arr_y)

        self.assertEqual(x.inner(y), x_fac.inner(y_fac))
        self.assertEqual(x.inner(y_fac), x_fac.inner(y))


if __name__ == '__main__':
    unittest.main(exit=False)
