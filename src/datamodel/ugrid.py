# -*- coding: utf-8 -*-
"""
ugrid.py -- n-dimensional uniform grid

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object

import numpy as np
from copy import deepcopy

from RL.utility.utility import InputValidationError
from RL.datamodel.coord import Coord


def asugrid(obj):
    """Make a Ugrid out of `obj`. Minimum requirement is that `obj`
    has a `shape` attribute or can be converted to a NumPy array.
    If present, `center` and `spacing` are used as well. If `obj` is
    already a Ugrid, `obj` will be returned (no copy)."""

    if isinstance(obj, Ugrid):
        return obj
    else:
        return ugrid(obj)


def ugrid(obj):
    """Return a Ugrid generated from `obj`. Minimum requirement is that `obj`
    has a `shape` attribute or can be converted to a NumPy array.
    If present, `center` and `spacing` are used as well. This function
    will always make a copy."""

    shape = getattr(obj, 'shape', np.asarray(obj).shape)
    center = getattr(obj, 'center', None)
    spacing = getattr(obj, 'spacing', None)

    return Ugrid(shape, center, spacing)


class Ugrid(object):
    """Uniform grid class.
    TODO: finish this
    """

    # TODO: add @convert and @check decorators
    def __init__(self, shape, center=None, spacing=None):

        # Convert values, apply defaults, check consistency, assign

        shape = np.array(shape)
        if not shape.ndim == 1:
            raise InputValidationError(shape.ndim, 1, 'shape.ndim')
        self._shape = shape
        dim = len(shape)

        # `None` implies all zeros
        center = np.zeros(dim) if center is None else np.array(center)
        if not center.shape == (dim,):
            raise InputValidationError(center.shape, (dim,), 'center.shape')
        self._center = center

        # `None` implies all ones
        if spacing is None:
            spacing = np.ones(dim)
        else:
            try:  # single value is broadcast
                spacing = float(spacing)
                spacing = spacing * np.ones(dim)
            except TypeError:  # spacing is an array - test for correct shape
                spacing = np.array(spacing)
                if not spacing.shape == (dim,):
                    raise InputValidationError(spacing.shape, dim,
                                               'spacing.shape')

        if not np.all(spacing > 0):
            raise ValueError('`spacing` must be all positive.')
        self._spacing = spacing

        # Initialize coords attribute; check if this is crucial for speed
        self._update_coord()

    # Elementary properties

    @property
    def shape(self):
        """The grid shape. Read-only (for now)."""
        return self._shape

    @property
    def center(self):
        """The grid center."""
        return self._center

    @center.setter
    def center(self, new_center):
        """Set the grid center. `None` means all zero."""
        if new_center is None:
            new_center = np.zeros(self.dim)
        else:
            new_center = np.array(new_center)
        if not new_center.shape == (self.dim,):
            raise InputValidationError(new_center.shape, (self.dim,),
                                       'new_center.shape')
        self._center = new_center
        self._update_coord()
        # Add code here to update depending cached properties

    @property
    def ref_point(self):
        return self.center

    @property
    def spacing(self):
        """The spacing between grid points."""
        return self._spacing

    @spacing.setter
    def spacing(self, new_spacing):
        """Set the grid spacing. `None` means all ones."""
        if new_spacing is None:
            self._spacing = np.ones(self.dim)
        else:
            new_spacing = np.array(new_spacing)
        if not new_spacing.shape == (self.dim,):
            raise InputValidationError(new_spacing.shape, (self.dim,),
                                       new_spacing.shape)
        self._spacing = new_spacing
        self._update_coord()
        # Add code here to update depending cached properties

    # Derived properties

    @property
    def dim(self):
        """Dimension of the grid"""
        return len(self.shape)

    @property
    def tsize(self):
        """Vector with total lengths of the grid (xmax - xmin)"""
        return self.spacing * self.shape

    @property
    def ntotal(self):
        """Total number of grid points"""
        return np.prod(self.shape)

    @property
    def xmin(self):
        """Vector with minimum coordinates"""
        return self.center - (self.shape - 1) / 2. * self.spacing

    @property
    def xmax(self):
        """Vector with maximum coordinates"""
        return self.center + (self.shape - 1) / 2. * self.spacing

    # Public methods

    def copy(self):
        """Return a (deep) copy."""
        return deepcopy(self)

    def is_subgrid(self, other):
        other = asugrid(other)

        if np.any(self.shape > other.shape):
            return False

        if np.any(self.xmin < other.xmin) or np.any(self.xmax > other.xmax):
            return False

        # other.spacing should be an integer multiple of self.spacing
        spc_factors = other.spacing / self.spacing
        if any(spc_factors - np.asarray(spc_factors, dtype=int) != 0):
            return False

    # Private methods

    def _update_coord(self):
        coord_vecs = [self._center[i]
                      + (np.arange(self._shape[i]) - (self._shape[i] - 1) / 2)
                      * self._spacing[i]
                      for i in range(self.dim)]
        self.coord = Coord(*coord_vecs)

    # Magic methods

    def __eq__(self, other):
        """For now, check equality of all essential attributes.
        TODO: maybe check for equality 'up to epsilon' instead (numpy.close).
        """

        return (np.all(self.shape == other.shape) and
                np.all(self.center == other.center) and
                np.all(self.spacing == other.spacing))
