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

import doctest
from doctest import IGNORE_EXCEPTION_DETAIL, ELLIPSIS, NORMALIZE_WHITESPACE
import os

import odl

root_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'source')
test_dirs = ['guide']
exclude_files = ['faq.rst']

doc_src_files = []
doctest_optionflags = NORMALIZE_WHITESPACE | ELLIPSIS | IGNORE_EXCEPTION_DETAIL

for test_dir in test_dirs:
    for path, _, filenames in os.walk(os.path.join(root_dir, test_dir)):
        for filename in filenames:
            if ((filename.endswith('.rst') or filename.endswith('.py')) and
                    filename not in exclude_files):
                doc_src_files.append(path + '/' + filename)

for doc_src_file in doc_src_files:
    doctest.testfile(doc_src_file, module_relative=False, report=True,
                     extraglobs={'odl': odl},
                     optionflags=doctest_optionflags)
