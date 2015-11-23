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
from builtins import str, super

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
from odl.operator.utility import matrix_representation
from odl.util.testutils import almost_equal


class MultiplyOp(odl.Operator):

    """Multiply with matrix.
    """

    def __init__(self, matrix, domain=None, range=None):
        domain = (odl.Rn(matrix.shape[1])
                  if domain is None else domain)
        range = (odl.Rn(matrix.shape[0])
                 if range is None else range)
        self.matrix = matrix

        super().__init__(domain, range, linear=True)

    def _apply(self, rhs, out):
        np.dot(self.matrix, rhs.data, out=out.data)

    @property
    def adjoint(self):
        return MultiplyOp(self.matrix.T, self.range, self.domain)


def test_matrix_representation():
    # Verify that the matrix representation function returns the correct matrix

    A = np.random.rand(3, 3)

    Aop = MultiplyOp(A)

    the_matrix = matrix_representation(Aop)

    assert almost_equal(np.sum(np.abs(A - the_matrix)), 1e-6)


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
