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
from RL.space.set import IntervProd, Interval, Rectangle
from RL.utility.testutils import RLTestCase

standard_library.install_aliases()

def getRandomPoint(set):
    if instanceof(set, IntervProd):
        return np.random.rand(set.dim) * (set.end - set.begin) + set.begin
    else:
        raise NotImplementedException("unknown type")


class IntervProdTest(RLTestCase):
    def testInit(self):
        set = IntervProd(1, 2)
        set = IntervProd([1], [2])
        set = IntervProd((1,), (2,))

        set = IntervProd([1, 2, 3], [4, 5, 6])
        set = IntervProd((1, 2, 3), (4, 5, 6))

        set = IntervProd((1, 2, 3), (1, 2, 3))

        with self.assertRaises(ValueError):
            set = IntervProd(2, 1)

        with self.assertRaises(ValueError):
            set = IntervProd((1, 2, 3), (1, 2, 0))

    def testBegin(self):
        set = IntervProd(1, 2)
        self.assertAlmostEquals(set.begin, 1)

        set = IntervProd([1], [2])
        self.assertAlmostEquals(set.begin, 1)

        set = IntervProd([1,2,3], [5,6,7])
        self.assertAllAlmostEquals(set.begin, [1,2,3])

    def testEnd(self):
        set = IntervProd(1, 2)
        self.assertAlmostEquals(set.end, 2)

        set = IntervProd([1], [2])
        self.assertAlmostEquals(set.end, 2)

        set = IntervProd([1,2,3], [5,6,7])
        self.assertAllAlmostEquals(set.end, [5,6,7])

    def testDim(self):
        set = IntervProd(1, 2)
        self.assertEquals(set.dim, 1)

        set = IntervProd(1, 1)
        self.assertEquals(set.dim, 1)

        set = IntervProd([1], [2])
        self.assertEquals(set.dim, 1)

        set = IntervProd([1,2,3], [5,6,7])
        self.assertEquals(set.dim, 3)

        set = IntervProd([1,2,3], [1,6,7])
        self.assertEquals(set.dim, 3)

    def testTrueDim(self):
        set = IntervProd(1, 2)
        self.assertEquals(set.truedim, 1)

        set = IntervProd(1, 1)
        self.assertEquals(set.truedim, 0)

        set = IntervProd([1], [2])
        self.assertEquals(set.truedim, 1)

        set = IntervProd([1,2,3], [5,6,7])
        self.assertEquals(set.truedim, 3)

        set = IntervProd([1,2,3], [1,6,7])
        self.assertEquals(set.truedim, 2)

    def testVolume(self):
        set = IntervProd(1, 2)
        self.assertEquals(set.dim, 2-1)

        set = IntervProd([1,2,3], [5,6,7])
        self.assertAlmostEquals(set.volume, (5-1)*(6-2)*(7-3))

    def testEquals(self):
        interval1 = IntervProd(1, 2)
        interval2 = IntervProd(1, 2)
        interval3 = IntervProd([1], [2])
        interval4 = IntervProd(2, 3)
        rectangle1 = IntervProd([1, 2], [2, 3])
        rectangle2 = IntervProd((1, 2), (2, 3))
        rectangle3 = IntervProd([0, 2], [2, 3])

        self.assertTrue(interval1.equals(interval1))
        self.assertTrue(interval1.equals(interval2))
        self.assertTrue(interval1.equals(interval3))
        self.assertFalse(interval1.equals(interval4))
        self.assertFalse(interval1.equals(rectangle1))
        self.assertTrue(rectangle1.equals(rectangle1))
        self.assertTrue(rectangle2.equals(rectangle2))
        self.assertFalse(rectangle1.equals(rectangle3))

        #Test operators
        self.assertTrue(interval1 == interval1)
        self.assertFalse(interval1 != interval1)
       
        self.assertFalse(interval1 == interval4)
        self.assertTrue(interval1 != interval4)
        
        self.assertFalse(interval1 == rectangle1)
        self.assertTrue(interval1 != rectangle1)

        self.assertTrue(rectangle1 == rectangle1)
        self.assertFalse(rectangle1 != rectangle1)
        
        self.assertFalse(rectangle1 == rectangle3)
        self.assertTrue(rectangle1 != rectangle3)

    def testContains(self):
        set1 = IntervProd(1, 2)
        set2 = IntervProd(1, 2)

        self.assertTrue(set1.equals(set2))
        self.assertTrue(set1 == set2)
        self.assertFalse(set1 != set2)
        
class IntervalTest(RLTestCase):
    def testInit(self):
        set = Interval(1, 2)
        set = Interval([1], [2])

        with self.assertRaises(ValueError):
            set = Interval([1, 2], [3, 4])

    def testLength(self):
        set = Interval(1, 2)
        self.assertEquals(set.length, set.volume)
        self.assertEquals(set.length, 1)


class RectangleTest(RLTestCase):
    def testInit(self):
        set = Rectangle([1, 2], [2, 3])

        with self.assertRaises(ValueError):
            set = Rectangle(1, 2)

        with self.assertRaises(ValueError):
            set = Rectangle([1, 2, 3], [4, 5, 6])

    def testArea(self):
        set = Rectangle([1, 2], [3, 4])
        self.assertEquals(set.area, set.volume)
        self.assertEquals(set.area, (3-1)*(4-2))

if __name__ == '__main__':
    unittest.main(exit=False)
