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

"""Run all doctests in the online documentation.

Running this file causes all relevant files in the online documentation to
be run by ``doctest``, and any exception will give a FAILED result.
All other results are considered PASSED.

This test file assumes that all dependencies are installed.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import doctest
from doctest import IGNORE_EXCEPTION_DETAIL, ELLIPSIS, NORMALIZE_WHITESPACE
import os
import pytest
try:
    import matplotlib
    matplotlib.use('Agg')  # To avoid the backend freezing
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Modules to be added to testing globals
import numpy
import scipy
import odl
try:
    import proximal
except ImportError:
    proximal = None

doctest_extraglobs = {'odl': odl, 'np': numpy, 'scipy': scipy,
                      'proximal': proximal}

root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        '../../doc/source')
root_dir = os.path.normpath(root_dir)
test_dirs = ['guide', 'getting_started']
test_suffixes = ['.rst', '.py']
exclude_files = ['faq.rst']

doc_src_files = []
doctest_optionflags = NORMALIZE_WHITESPACE | ELLIPSIS | IGNORE_EXCEPTION_DETAIL

for test_dir in test_dirs:
    for path, _, filenames in os.walk(os.path.join(root_dir, test_dir)):
        for filename in filenames:
            if (any(filename.endswith(suffix) for suffix in test_suffixes) and
                    filename not in exclude_files):
                doc_src_files.append(os.path.join(path, filename))


@pytest.fixture(scope="module", ids=doc_src_files, params=doc_src_files)
def doc_src_file(request):
    return request.param


@pytest.mark.skipif("not pytest.config.getoption('--doctest-doc')",
                    reason='Need --doctest-doc option to run')
def test_file(doc_src_file):
    doctest.testfile(doc_src_file, module_relative=False, report=True,
                     extraglobs=doctest_extraglobs, verbose=True,
                     optionflags=doctest_optionflags)
    plt.close('all')


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v', '--doctest-doc'])
