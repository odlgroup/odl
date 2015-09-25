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

# ODL imports
from odl.util.testutils import ODLTestCase


class ImportStarTest(ODLTestCase):
    def test_all(self):
        import odl
        # Create Cn
        C1 = odl.Cn(3)
        C2 = odl.space.Cn(3)
        C3 = odl.space.ntuples.Cn(3)

        # Three ways of creating the identity
        I1 = odl.IdentityOperator(C3)
        I2 = odl.operator.IdentityOperator(C3)
        I3 = odl.operator.default_ops.IdentityOperator(C3)

        # Test that utility needs to be explicitly imported
        x = odl.util.utility.array1d_repr
        with self.assertRaises(AttributeError):
            x = odl.array1d_repr

if __name__ == '__main__':
    unittest.main(exit=False)
