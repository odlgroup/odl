# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Numpy functions not available in the minimal required version."""

from __future__ import print_function, division, absolute_import
import numpy as np


__all__ = ('moveaxis', 'flip')


# TODO: Remove when Numpy 1.11 is an ODL dependency
def moveaxis(a, source, destination):
    """Move axes of an array to new positions.

    Other axes remain in their original order.

    This function is a backport of `numpy.moveaxis` introduced in
    NumPy 1.11.

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


# TODO: Remove when Numpy 1.12 is an ODL dependency
def flip(a, axis):
    """Reverse the order of elements in an array along the given axis.

    This function is a backport of `numpy.flip` introduced in NumPy 1.12.

    See Also
    --------
    numpy.flip
    """
    if not hasattr(a, 'ndim'):
        a = np.asarray(a)
    indexer = [slice(None)] * a.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError('axis={} is invalid for the {}-dimensional input '
                         'array'.format(axis, a.ndim))
    return a[tuple(indexer)]


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
