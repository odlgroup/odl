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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External module imports
import pytest


def test_all_imports():
    import odl
    # Create Cn
    odl.cn(3)
    odl.space.cn(3)
    C3 = odl.space.space_utils.cn(3)

    # Three ways of creating the identity
    odl.IdentityOperator(C3)
    odl.operator.IdentityOperator(C3)
    odl.operator.default_ops.IdentityOperator(C3)

    # Test that utility needs to be explicitly imported
    odl.util.utility.array1d_repr
    with pytest.raises(AttributeError):
        odl.array1d_repr

if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
