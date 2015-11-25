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
import numpy as np

# ODL imports
import odl
from odl.operator.operator_utilities import matrix_representation
from odl.space.ntuples import MatVecOperator
from odl.util.testutils import almost_equal


def test_matrix_representation():
    # Verify that the matrix representation function returns the correct matrix

    n = 3
    rn = odl.Rn(n)
    A = np.random.rand(n, n)

    Aop = MatVecOperator(rn, rn, A) #MultiplyOp(A)

    the_matrix = matrix_representation(Aop)

    assert almost_equal(np.sum(np.abs(A - the_matrix)), 1e-6)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
