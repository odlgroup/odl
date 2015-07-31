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

"""
Utilities for use inside the ODL project, not for external use.
"""

# Imports for common Python 2/3 codebase
from __future__ import (division, unicode_literals, print_function,
                        absolute_import)
from future import standard_library

# External module imports
from textwrap import dedent, fill

standard_library.install_aliases()


def errfmt(errstr):
    return fill(dedent(errstr)).lstrip()


def array1d_repr(array):
    if len(array) < 7:
        return repr(array[:].tolist())
    else:
        return (repr(array[:3].tolist()).rstrip(']') + ', ..., ' +
                repr(array[-3:].tolist()).strip('['))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
