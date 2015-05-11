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

# RL imports
from RL.space.space import *
from RL.space.euclidean import *
from RL.space.function import *
from RL.space.product import *
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class ProductTest(RLTestCase):
    def testRxR(self):
        H = RN(2)
        HxH = ProductSpace(H, H)
        self.assertTrue(len(HxH) == 2)

        v1 = H.makeVector([1, 2])
        v2 = H.makeVector([3, 4])
        v = HxH.makeVector(v1, v2)
        u = HxH.makeVector([1, 2], [3, 4])

        self.assertAllAlmostEquals([v1, v2], v)
        self.assertAllAlmostEquals([v1, v2], u)

    def testLinComb(self):
        H = RN(2)
        HxH = ProductSpace(H, H)

        v1 = H.makeVector([1, 2])
        v2 = H.makeVector([5, 3])
        u1 = H.makeVector([-1, 7])
        u2 = H.makeVector([2, 1])

        v = HxH.makeVector(v1, v2)
        u = HxH.makeVector(u1, u2)
        z = HxH.empty()

        a = 3.12
        b = 1.23

        expected = [a*v1 + b*u1, a*v2 + b*u2]
        HxH.linComb(z, a, v, b, u)

        self.assertAllAlmostEquals(z, expected)

    def testNorm(self):
        H = EuclideanSpace(2)
        v1 = H.makeVector([1, 2])
        v2 = H.makeVector([5, 3])

        # 0-norm
        HxH = NormedProductSpace(H, H, ord=0.0)
        w = HxH.makeVector(v1, v2)
        self.assertAlmostEquals(w.norm(), 2)  # No term is nonzero

        # 1-norm
        HxH = NormedProductSpace(H, H, ord=1.0)
        w = HxH.makeVector(v1, v2)
        self.assertAlmostEquals(w.norm(), v1.norm()+v2.norm())

        # 2-norm
        HxH = NormedProductSpace(H, H, ord=2.0)
        w = HxH.makeVector(v1, v2)
        self.assertAlmostEquals(w.norm(), (v1.norm()**2+v2.norm()**2)**(1/2.0))

        # -inf norm
        HxH = NormedProductSpace(H, H, ord=-float('inf'))
        w = HxH.makeVector(v1, v2)
        self.assertAlmostEquals(w.norm(), min(v1.norm(), v2.norm()))

        # inf norm
        HxH = NormedProductSpace(H, H, ord=float('inf'))
        w = HxH.makeVector(v1, v2)
        self.assertAlmostEquals(w.norm(), max(v1.norm(), v2.norm()))


class PowerTest(RLTestCase):
    def testRxR(self):
        H = RN(2)
        HxH = PowerSpace(H, 2)
        self.assertTrue(len(HxH) == 2)

        v1 = H.makeVector([1, 2])
        v2 = H.makeVector([3, 4])
        v = HxH.makeVector(v1, v2)
        u = HxH.makeVector([1, 2], [3, 4])

        self.assertAllAlmostEquals([v1, v2], v)
        self.assertAllAlmostEquals([v1, v2], u)

    def testLinComb(self):
        H = RN(2)
        HxH = PowerSpace(H, 2)

        v1 = H.makeVector([1, 2])
        v2 = H.makeVector([5, 3])
        u1 = H.makeVector([-1, 7])
        u2 = H.makeVector([2, 1])

        v = HxH.makeVector(v1, v2)
        u = HxH.makeVector(u1, u2)
        z = HxH.empty()

        a = 3.12
        b = 1.23

        expected = [a*v1 + b*u1, a*v2 + b*u2]
        HxH.linComb(z, a, v, b, u)

        self.assertAllAlmostEquals(z, expected)

    def testInplaceModify(self):
        H = RN(2)
        HxH = PowerSpace(H, 2)

        v1 = H.makeVector([1, 2])
        v2 = H.makeVector([5, 3])
        u1 = H.makeVector([-1, 7])
        u2 = H.makeVector([2, 1])
        z1 = H.empty()
        z2 = H.empty()

        v = HxH.makeVector(v1, v2)
        u = HxH.makeVector(u1, u2)
        z = HxH.makeVector(z1, z2)  # z is simply a wrapper for z1 and z2

        a = 3.12
        b = 1.23

        HxH.linComb(z, a, v, b, u)

        # Assert that z1 and z2 has been modified as well
        self.assertAllAlmostEquals(z, [z1, z2])


if __name__ == '__main__':
    unittest.main(exit=False)
