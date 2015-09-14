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

from future import standard_library
standard_library.install_aliases()

# External module imports
import unittest
from math import pi, sqrt
import numpy as np

# ODL imports
from odl.space.default import L2
from odl.space.domain import Interval, Rectangle
from odl.utility.testutils import ODLTestCase


#class L2Test(ODLTestCase):
#    def test_interval(self):
#        I = Interval(0, pi)
#        l2 = L2(I)
#        l2sin = l2.element(np.sin)
#
#        rn = En(10)
#        d = uniform_discretization(l2, rn)
#
#        sind = d.element(l2sin)
#
#        self.assertAlmostEqual(sind.norm(), sqrt(pi/2))
#
#    def test_rectangle(self):
#        R = Rectangle((0, 0), (pi, 2*pi))
#        l2 = L2(R)
#        l2sin = l2.element(lambda p: np.sin(p[0]) * np.sin(p[1]))
#
#        n = 10
#        m = 10
#        rn = En(n*m)
#        d = uniform_discretization(l2, rn, (n, m))
#
#        sind = d.element(l2sin)
#
#        self.assertAlmostEqual(sind.norm(), sqrt(pi**2 / 2))

if __name__ == '__main__':
    unittest.main(exit=False)
