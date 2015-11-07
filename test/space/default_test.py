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
import pytest
from math import pi, sqrt
import numpy as np

# ODL imports
import odl
from odl import L2
from odl.util.testutils import almost_equal


def test_interval():
    l2space = L2(odl.Interval(0, pi))
    l2discr = odl.uniform_discr(l2space, 10)

    l2sin = l2space.element(np.sin)
    discr_sin = l2discr.element(l2sin)

    assert almost_equal(discr_sin.norm(), sqrt(pi/2))

def test_rectangle():
    l2space = L2(odl.Rectangle((0, 0), (pi, 2*pi)))
    n, m = 10, 10
    l2discr = odl.uniform_discr(l2space, (n, m))

    l2sin2 = l2space.element(lambda x, y: np.sin(x) * np.sin(y))
    discr_sin2 = l2discr.element(l2sin2)

    assert almost_equal(discr_sin2.norm(), pi / sqrt(2))
    
def test_addition():
    l2space = L2(odl.Interval(0, pi))
    l2sin = l2space.element(np.sin)
    l2cos = l2space.element(np.cos)
    
    sum_func = l2sin + l2cos

    for x in [0.0, 0.2, 1.0]:
        assert almost_equal(sum_func(x), np.sin(x) + np.cos(x))

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\','/') + ' -v'))
