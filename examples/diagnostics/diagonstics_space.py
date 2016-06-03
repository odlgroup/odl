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

"""Run the standardized diagonstics suite on some of the spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import odl

print('\n\n TESTING FOR Lp SPACE \n\n')

discr = odl.uniform_discr(0, 1, 10)
odl.diagnostics.SpaceTest(discr).run_tests()

print('\n\n TESTING FOR Rn SPACE \n\n')

spc = odl.rn(10)
odl.diagnostics.SpaceTest(spc).run_tests()


print('\n\n TESTING FOR Cn SPACE \n\n')

spc = odl.cn(10)
odl.diagnostics.SpaceTest(spc).run_tests()


if 'cuda' in odl.FN_IMPLS:
    print('\n\n TESTING FOR CUDA Rn SPACE \n\n')

    spc = odl.rn(10, impl='cuda')
    odl.diagnostics.SpaceTest(spc, eps=0.0001).run_tests()
