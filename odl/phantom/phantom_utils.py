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

"""Utilities for creating phantoms."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('cylinders_from_ellipses',)


def cylinders_from_ellipses(ellipses2d):
    """Create 3d cylinders from ellipses."""
    ellipses2d = np.asarray(ellipses2d)
    ellipses3d = np.zeros((ellipses2d.shape[0], 10))
    ellipses3d[:, [0, 1, 2, 4, 5, 7]] = ellipses2d
    ellipses3d[:, 3] = 100000.0

    return ellipses3d


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
