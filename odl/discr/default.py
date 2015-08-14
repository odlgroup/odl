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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
from future import standard_library
standard_library.install_aliases()

from odl.discr.discretization import Discretization
from odl.discr.grid import TensorGrid
from odl.discr.operators import GridCollocation, NearestInterpolation
from odl.space.cartesian import Ntuples
from odl.space.function import FunctionSet
from odl.utility.utility import errfmt


class RawNearestInterpDiscretization(Discretization):

    """Discretization based on nearest neighbor interpolation."""

    def __init__(self, ip_funcset, grid, ntuples):
        """Initialize a new instance.

        Parameters
        ----------
        ip_funcset : `FunctionSet`
            Set of functions on an `IntervalProd`. The operator range.
        grid : `TensorGrid`
            The grid on which to interpolate. Must be contained in
            `ip_funcset.domain`.
        ntuples : `Ntuples`
            An implementation of n-tuples. The operator domain.
        """
        if not isinstance(ip_funcset, FunctionSet):
            raise TypeError(errfmt('''
            `ip_funcset` {} not an `FunctionSet` instance.
            '''.format(ip_funcset)))

        if not isinstance(grid, TensorGrid):
            raise TypeError(errfmt('''
            `grid` {} not a `TensorGrid` instance.
            '''.format(grid)))

        if not isinstance(ntuples, Ntuples):
            raise TypeError(errfmt('''
            `ntuples` {} not an `Ntuples` instance.'''.format(ntuples)))

        if grid.ntotal != ntuples.dim:
            raise ValueError(errfmt('''
            total number {} of grid points not equal to `ntuples.dim`
            {}.'''.format(grid.ntotal, ntuples.dim)))

        restr = GridCollocation(ip_funcset, grid, ntuples)
        ext = NearestInterpolation(ip_funcset, grid, ntuples)
        super().__init__(ip_funcset, ntuples, restr, ext)
