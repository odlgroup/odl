# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Test that all examples run.

Running this file causes all examples to be run, and any exception will give
a FAILED result. All other results are considered PASSED.

Plots are displayed while the examples are run, but are closed after each
example. A user can, if he/she wants inspect the plots to further see if the
examples are OK.

This package assumes that all dependencies are installed.
"""

from __future__ import division
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
