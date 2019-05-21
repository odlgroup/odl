# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Numpy functions not available in the minimal required version."""

from __future__ import print_function, division, absolute_import
import numpy as np


__all__ = ('moveaxis', 'flip', 'roll')


# --- Numpy 1.11 --- #


def moveaxis(a, source, destination):
    """Move axes of an array to new positions.

    Other axes remain in their original order.

    This function is a backport of `numpy.moveaxis` introduced in
    NumPy 1.11.

    See Also
    --------
    numpy.moveaxis
    """
    major, minor, _ = [int(s) for s in np.version.short_version.split('.')]
    if (major, minor) >= (1, 11):
        return np.moveaxis(a, source, destination)
    else:
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


# --- Numpy 1.12 --- #


def flip(a, axis):
    """Reverse the order of elements in an array along the given axis.

    This function is a backport of `numpy.flip` introduced in NumPy 1.12.

    See Also
    --------
    numpy.flip
    """
    major, minor, _ = [int(s) for s in np.version.short_version.split('.')]
    if (major, minor) >= (1, 12):
        return np.flip(a, axis)
    else:
        if not hasattr(a, 'ndim'):
            a = np.asarray(a)
        indexer = [slice(None)] * a.ndim
        try:
            indexer[axis] = slice(None, None, -1)
        except IndexError:
            raise ValueError('axis={} is invalid for the {}-dimensional input '
                             'array'.format(axis, a.ndim))
        return a[tuple(indexer)]


# --- Numpy 1.13 --- #


def roll(a, shift, axis=None):
    """Roll array elements along a given axis.

    Elements that roll beyond the last position are re-introduced at
    the first.

    This function is a backport of `numpy.roll` introduced in NumPy 1.13.

    See Also
    --------
    numpy.roll
    """
    major, minor, _ = [int(s) for s in np.version.short_version.split('.')]
    if (major, minor) >= (1, 13):
        return np.roll(a, shift, axis)
    else:
        if axis is None:
            return roll(a.ravel(), shift, 0).reshape(a.shape)
        elif np.isscalar(axis):
            return np.roll(a, shift, axis)
        else:
            axis = tuple(axis)
            if axis == ():
                return a.copy()

            if np.isscalar(shift):
                shift = [shift] * len(axis)
            elif len(shift) != len(axis):
                raise ValueError('`shift` must be integer or of the same '
                                 'length as `axis`')

            rolled = np.roll(a, shift[0], axis[0])
            for sh, ax in zip(shift[1:], axis[1:]):
                rolled = np.roll(rolled, sh, ax)

            return rolled


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
