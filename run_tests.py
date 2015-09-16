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

"""Use nose to find all tests in the 'test' folder and run them."""

from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

import nose
import sys
from run_doctests import run_doctests


def run_tests():
    arg = sys.argv[:1]
    arg.append('--verbosity=2')
    arg.append('--with-coverage')
    arg.append('--cover-package=odl')
    out = nose.run(defaultTest='./test/.', argv=arg)

if __name__ == '__main__':
    run_doctests()
    run_tests()
