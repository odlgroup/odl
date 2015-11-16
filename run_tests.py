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

"""Use pytest to find all tests in the 'test' folder and all doctests in
'odl' and run them."""

from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

import sys
import pytest
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False


if __name__ == '__main__':
    from distutils.version import StrictVersion
    from warnings import warn

    arg = sys.argv[:1]
    arg.append('./test/')

    # Use doctests only for pytest >= 2.7.0, otherwise we get whitespace errors
    if StrictVersion(pytest.__version__) >= '2.7.0':
        arg.append('./odl/')
        arg.append('--doctest-modules')
    else:
        warn('Ignoring doctests due to deprecated pytest version {}. '
             'Required support for doctest option flags was added in 2.7.0.'
             ''.format(pytest.__version__))
    if COVERAGE_AVAILABLE:
        cov = coverage.Coverage()
        cov.start()

    from odl import CUDA_AVAILABLE
    if not CUDA_AVAILABLE:
        arg.append('--ignore=odl/space/cu_ntuples.py')

    result = pytest.main(arg)

    if COVERAGE_AVAILABLE:
        cov.stop()
        cov.save()
        cov.html_report()
    
    if result != 0:
        sys.exit(1)
