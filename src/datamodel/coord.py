# -*- coding: utf-8 -*-
"""
coord.py -- coordinates for n-dimensional rectangular grids

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

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import range
from builtins import object
from future import standard_library
standard_library.install_aliases()
from future.builtins import range
from future.builtins import object

import numpy as np
from RL.utility.utility import flat_tuple, vec_list_from_arg


class Coord(object):

    def __init__(self, *vectors):

        vecs = vec_list_from_arg(vectors)
        self._vecs = tuple(vecs)

    @property
    def vecs(self):
        return self._vecs

    @vecs.setter
    def vecs(self, *new_vecs):

        self.__init__(new_vecs)

    @property
    def dim(self):
        return len(self._vecs)

    @property
    def ntotal(self):
        return np.prod([len(vec) for vec in self.vecs])

    def asarr(self, slc=None):
        """Return matrix of all possible coordinates (x-fastest ordering).
        If `slc` is a slice object or a list of slice objects, the
        coordinate vectors are sliced accordingly."""
        if slc is None:
            slicedvecs = self.vecs
        else:
            if isinstance(slc, slice):
                slc = np.index_exp[slc]
            else:
                slc = flat_tuple(slc)

            slicedvecs = []
            for i in range(self.dim):
                slicedvecs.append(self.vecs[i][slc[i]])

        ntotal = np.prod([len(vec) for vec in slicedvecs])
        arr = np.empty((ntotal, self.dim))
        repeats = 1
        for i in range(self.dim):
            repeated = np.repeat(slicedvecs[i], repeats)
            arr[:, i] = np.resize(repeated, ntotal)
            repeats *= len(slicedvecs[i])

        return arr

    def __getitem__(self, slc):
        if isinstance(slc, slice):
            slc = np.index_exp[slc]
        else:
            slc = flat_tuple(slc)

        if len(slc) < self.dim:
            raise IndexError('too few indices for coords')
        elif len(slc) > self.dim:
            raise IndexError('too many indices for coords')

        slicedvecs = []
        for i in range(self.dim):
            slicedvecs.append(self.vecs[i][slc[i]])

        return Coord(*slicedvecs)


class CoordTransform(object):
    """Base class for coordinate transforms. The `mapping` must accept
    either an array (column-wise) or a list of coordinate vectors as arguments
    and return an array of transformed vectors.
    The transform is executed by calling the class object and returns the
    transformed vector array.
    Subclasses can customize the initialization of `mapping` for special
    cases.
    """

    def __init__(self, mapping):
        # TODO: inspect mapping signature
        self.mapping = mapping

    def __call__(self, coo):
        coo_arr = coo.asarr()
        try:
            return self.mapping(coo_arr)
        except TypeError:
            return self.mapping(*coo_arr.T)


class CoordTransformWarpedGraph(CoordTransform):
    """Coordinate transform for a warped function graph. The `function` must
    accept either an array (column-wise) or a list of coordinate vectors as
    arguments and return a column of the same length (one value for each row).
    The `warp_graph` mapping must accept the same vector array or list of
    vectors, extended by the column from `function`, and return an array of
    the same shape.
    The transform is executed by calling the class object and returns the
    final array.
    """
    def __init__(self, function, graph_warp=None):

        def mapping(coo_arr):
            graph = np.empty(coo_arr.shape + np.array((0, 1)))
            graph[:, :-1] = coo_arr
            try:
                graph[:, -1] = function(coo_arr)
            except TypeError:
                graph[:, -1] = function(*coo_arr.T)

            if graph_warp:
                try:
                    graph = graph_warp(graph)
                except TypeError:
                    graph = graph_warp(*graph.T)

            return graph

        self.mapping = mapping
