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

"""Test that all examples run.

Running this file causes all examples to be run, and any exception will give
a FAILED result. All other results are considered PASSED.

Plots are displayed while the examples are run, but are closed after each
example. A user can, if he/she wants inspect the plots to further see if the
examples are OK.

This package assumes that all dependencies are installed.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import os
import imp
import pytest
try:
    import matplotlib
    matplotlib.use('Agg')  # To avoid the backend freezing
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Make a fixture for all examples
this_file_path = os.path.dirname(os.path.abspath(__file__))
examples_path = os.path.join(this_file_path,
                             os.path.pardir, os.path.pardir, 'examples')
example_ids = []
example_params = []
for dirpath, dirnames, filenames in os.walk(examples_path):
    for filename in [f for f in filenames if f.endswith(".py") and
                     not f.startswith('__init__')]:
        example_params.append(os.path.join(dirpath, filename))
        example_ids.append(filename[:-3])  # skip .py


@pytest.fixture(scope="module", ids=example_ids, params=example_params)
def example(request):
    return request.param


@pytest.mark.skipif("not pytest.config.getoption('--examples')",
                    reason='Need --examples option to run')
def test_example(example):
    imp.load_source('tmp', example)
    plt.close('all')


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v', '--examples'])
