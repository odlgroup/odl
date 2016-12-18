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

"""Numpy functions not available in the minimal required version."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('broadcast_to', 'moveaxis')

# TODO: Remove when numpy 1.10 is an ODL dependency
def broadcast_to(array, shape):
    """Broadcast an array to a new shape.

    This implementation is needed since NumPy introduces this function
    in version 1.10, which ODL doesn't have as a dependency yet.

    See Also
    --------
    numpy.broadcast_to
    """
    array = np.asarray(array)
    try:
        return np.broadcast_to(array, shape)
    except AttributeError:
        # The above requires numpy 1.10, fallback impl else
        shape = [m if n == 1 and m != 1 else 1
                 for n, m in zip(array.shape, shape)]
        return array + np.zeros(shape, dtype=array.dtype)


# TODO: Remove when numpy 1.11 is an ODL dependency
def moveaxis(a, source, destination):
    """Move axes of an array to new positions.

    Other axes remain in their original order.

    This implementation is needed since NumPy introduces this function
    in version 1.11, which ODL doesn't have as a dependency yet.

    See Also
    --------
    numpy.moveaxis
    """
    import numpy
    if hasattr(numpy, 'moveaxis'):
        return numpy.moveaxis(a, source, destination)

    try:
        source = list(source)
    except TypeError:
        source = [source]
    try:
        destination = list(destination)
    except TypeError:
        destination = [destination]

    source = [ax + a.ndim if ax < 0 else ax for ax in source]
    destination = [ax + a.ndim if ax < 0 else ax for ax in destination]

    order = [n for n in range(a.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return a.transpose(order)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
