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

"""Utilities for internal use."""


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

# External module imports
import numpy as np


__all__ = ('is_valid_input_array', 'is_valid_input_meshgrid',
           'meshgrid_input_order', 'vecs_from_meshgrid',
           'out_shape_from_meshgrid', 'out_shape_from_array')


def is_valid_input_array(x, d):
    """Test whether `x` is a correctly shaped array of points in R^d."""
    if not isinstance(x, np.ndarray):
        return False
    if d == 1:
        return x.ndim == 1 or x.ndim == 2 and x.shape[0] == 1
    else:
        return x.ndim == 2 and x.shape[0] == d


def is_valid_input_meshgrid(x, d):
    """Test whether `x` is a meshgrid sequence for points in R^d."""
    if isinstance(x, np.ndarray):
        return False
    if d > 1:
        try:
            np.broadcast(*x)
        except ValueError:  # cannot be broadcast
            return False

    return (len(x) == d and
            all(isinstance(xi, np.ndarray) for xi in x) and
            all(xi.ndim == d for xi in x))


def meshgrid_input_order(x):
    """Determine the ordering of a meshgrid argument."""
    # Case 1: all elements have the same shape -> non-sparse
    if all(xi.shape == x[0].shape for xi in x):
        # Contiguity check only works for meshgrid created with copy=True.
        # Otherwise, there is no way to find out the intended ordering.
        if all(xi.flags.f_contiguous for xi in x):
            return 'F'
        else:
            return 'C'
    # Case 2: sparse meshgrid, each member's shape has at most one non-one
    # entry (corner case of all ones is included)
    elif all(xi.shape.count(1) >= len(x) - 1 for xi in x):
        # Reversed ordering of dimensions in the meshgrid tuple indicates
        # 'F' ordering intention
        if all(xi.shape[-1 - i] != 1 for i, xi in enumerate(x)):
            return 'F'
        else:
            return 'C'
    else:
        return 'C'


def vecs_from_meshgrid(mg, order):
    """Get the coordinate vectors from a meshgrid (as a tuple)."""
    vecs = []
    for ax in range(len(mg)):
        select = [0] * len(mg)
        if str(order).upper() == 'F':
            select[-ax] = np.s_[:]
        else:
            select[ax] = np.s_[:]
        vecs.append(mg[ax][select])
    return tuple(vecs)


def out_shape_from_meshgrid(mg):
    """Get the broadcast output shape from a meshgrid."""
    if len(mg) == 1:
        return (len(mg[0]),)
    else:
        return np.broadcast(*mg).shape


def out_shape_from_array(arr, ndim):
    """Get the output shape from an array for ``ndim`` dimensions."""
    if arr.ndim == 1:
        return arr.shape
    else:
        return (arr.shape[1],)
