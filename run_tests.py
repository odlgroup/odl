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

"""Use pytest to find all tests in the 'test' folder and all doctests in 'odl' and run them."""

from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

import sys
import odl
import pytest
try:
    import pytest_cov
    PYTEST_COV_AVAILABLE = True
except ImportError:
    PYTEST_COV_AVAILABLE = False


if __name__ == '__main__':
    arg = sys.argv[:1]
    arg.append('./test/')
    arg.append('./odl/')
    arg.append('--doctest-modules')
    if PYTEST_COV_AVAILABLE:
        arg.append('--cov=odl')
        arg.append('--cov-report=term-missing')
    if not odl.CUDA_AVAILABLE:
        arg.append('--ignore=odl/space/cu_ntuples.py')

    pytest.main(arg)
