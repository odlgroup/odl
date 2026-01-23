# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run all doctests in the online documentation.

Running this file causes all relevant files in the online documentation to
be run by ``doctest``, and any exception will give a FAILED result.
All other results are considered PASSED.

This test file assumes that all dependencies are installed.
"""

from __future__ import division

import doctest
import os
from doctest import ELLIPSIS, IGNORE_EXCEPTION_DETAIL, NORMALIZE_WHITESPACE

import numpy
import pytest

import odl
from odl.core.util.testutils import simple_fixture

try:
    import matplotlib
    matplotlib.use('agg')  # prevent backend from freezing
    import matplotlib.pyplot as plt
except ImportError:
    pass


doctest_extraglobs = {'odl': odl, 'np': numpy}

root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        os.pardir, os.pardir, 'doc', 'source')
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


doc_src_file = simple_fixture("doc_src_file", doc_src_files)


@pytest.mark.suite("doc_doctests")
def test_file(doc_src_file):
    # FIXXXME: This doesn't seem to actually test the file :-(
    doctest.testfile(doc_src_file, module_relative=False, report=True,
                     extraglobs=doctest_extraglobs, verbose=True,
                     optionflags=doctest_optionflags)
    plt.close('all')


if __name__ == '__main__':
    odl.core.util.test_file(__file__)
