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

"""Utilities for internal use."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('normalized_index_expression',)


def normalized_index_expression(indices, shape, int_to_slice=False):
    """Enable indexing with almost Numpy-like capabilities.

    Implements the following features:

    - Usage of general slices and sequences of slices
    - Conversion of `Ellipsis` into an adequate number of ``slice(None)``
      objects
    - Fewer indices than axes by filling up with an `Ellipsis`
    - Error checking with respect to a given shape
    - Conversion of integer indices into corresponding slices

    Parameters
    ----------
    indices : `int`, `slice` or `sequence` of those
        Index expression to be normalized
    shape : `sequence` of `int`
        Target shape for error checking of out-of-bounds indices.
        Also needed to determine the number of axes.
    int_to_slice : `bool`, optional
        If `True`, turn integers into corresponding slice objects.

    Returns
    -------
    normalized : `tuple` of `slice`
        Normalized index expression
    """
    ndim = len(shape)
    # Support indexing with fewer indices as indexing along the first
    # corresponding axes. In the other cases, normalize the input.
    if np.isscalar(indices):
        indices = [indices, Ellipsis]
    elif (isinstance(indices, slice) or indices is Ellipsis):
        indices = [indices]

    indices = list(indices)
    if len(indices) < ndim and Ellipsis not in indices:
        indices.append(Ellipsis)

    # Turn Ellipsis into the correct number of slice(None)
    if Ellipsis in indices:
        if indices.count(Ellipsis) > 1:
            raise ValueError('cannot use more than one Ellipsis.')

        eidx = indices.index(Ellipsis)
        extra_dims = ndim - len(indices) + 1
        indices = (indices[:eidx] + [slice(None)] * extra_dims +
                   indices[eidx + 1:])

    # Turn single indices into length-1 slices if desired
    for (i, idx), n in zip(enumerate(indices), shape):
        if np.isscalar(idx):
            if idx < 0:
                idx += n

            if idx >= n:
                raise IndexError('Index {} is out of bounds for axis '
                                 '{} with size {}.'
                                 ''.format(idx, i, n))
            if int_to_slice:
                indices[i] = slice(idx, idx + 1)

    # Catch most common errors
    if any(s.start == s.stop and s.start is not None or
           s.start == n
           for s, n in zip(indices, shape) if isinstance(s, slice)):
        raise ValueError('Slices with empty axes not allowed.')
    if None in indices:
        raise ValueError('creating new axes is not supported.')
    if len(indices) > ndim:
        raise IndexError('too may indices: {} > {}.'
                         ''.format(len(indices), ndim))

    return tuple(indices)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
