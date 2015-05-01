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
from RL.space.product import ProductSpace  # , PowerSpace
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()


class ProductTest(RLTestCase):
    def testRxR(self):
        H = RN(2)
        HxH = ProductSpace(H, H)
        self.assertTrue(HxH.dimension == 2)

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

        v = HxH.makeVector(v1.copy(), v2.copy())
        u = HxH.makeVector(u1.copy(), u2.copy())
        z = HxH.empty()

        a = 3.12
        b = 1.23

        expected = [a*v1 + b*u1, a*v2 + b*u2]
        HxH.linComb(z, a, v, b, u)

        self.assertAllAlmostEquals(z, expected)


if __name__ == '__main__':
    unittest.main(exit=False)
