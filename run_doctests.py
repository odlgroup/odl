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

"""Run all doctests."""

from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

import nose
import sys


def run_doctests():
    arg = sys.argv[:1]
    arg.append('--verbosity=2')
    arg.append('--with-doctest')
    arg.append('--doctest-options=+NORMALIZE_WHITESPACE,+ELLIPSIS'
               ',+IGNORE_EXCEPTION_DETAIL')
    try:
        # pylint: disable=unused-import
        import odl.space.cuda
    except ImportError:
        arg.append('--ignore-files=cuda.py')
    out = nose.run(defaultTest='./odl/.', argv=arg)

if __name__ == '__main__':
    run_doctests()
