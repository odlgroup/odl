# Copyright 2014-2016 The ODL development group
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

"""Test for the Douglas-Rachford solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import odl

from odl.util.testutils import all_almost_equal


def test_l1():
    """Verify that the correct value is returned for l1 dist optimization."""

    # Define the space
    space = odl.rn(5)

    # Operator
    L = [odl.IdentityOperator(space)]

    # Data
    data_1 = odl.util.testutils.example_element(space)
    data_2 = odl.util.testutils.example_element(space)

    # Proximals
    prox_f = odl.solvers.proximal_l1(space, g=data_1)

    # Solve with f term dominating
    x = space.zero()
    prox_cc_g = [odl.solvers.proximal_cconj_l1(space, g=data_2, lam=0.5)]
    odl.solvers.douglas_rachford_pd(x, prox_f, prox_cc_g, L,
                                    tau=3.0, sigma=[1.0], niter=5)

    assert all_almost_equal(x, data_1, places=2)

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
