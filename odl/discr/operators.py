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
from future import standard_library
standard_library.install_aliases()

# External imports
import numpy as np
from scipy.interpolate import interpn

# ODL imports
from odl.discr.grid import TensorGrid
from odl.operator.operator import Operator
from odl.space.cartesian import Ntuples
from odl.space.function import FunctionSet
from odl.space.set import IntervalProd
from odl.utility.utility import errfmt


class GridCollocation(Operator):

    """Function evaluation at grid points.

    This is the default 'restriction' operator used by all core
    discretization classes.
    """

    def __init__(self, ip_funcset, grid, ntuples):
        """Initialize a new `PointCollocation` instance.

        Parameters
        ----------
        ip_funcset : `FunctionSet`
            Set of functions, the operator range. Its `domain` must
            be an `IntervalProduct`.
        grid : `TensorGrid`
            The grid on which to evaluate. Must be contained in
            `ip_funcset.domain`.
        ntuples : `Ntuples`
            Implementation of n-tuples, the operator domain. Its
            dimension must be equal to `grid.ntotal`.
        """
        if not isinstance(ip_funcset, FunctionSet):
            raise TypeError(errfmt('''
            `ip_funcset` {} not an `FunctionSet` instance.
            '''.format(ip_funcset)))

        if not isinstance(ip_funcset.domain, IntervalProd):
            raise TypeError(errfmt('''
            `domain` {} of `ip_funcset` not an `IntervalProd` instance.
            '''.format(ip_funcset.domain)))

        if not isinstance(grid, TensorGrid):
            raise TypeError(errfmt('''
            `grid` {} not a `TensorGrid` instance.
            '''.format(grid)))

        if not isinstance(ntuples, Ntuples):
            raise TypeError(errfmt('''
            `ntuples` {} not an `Ntuples` instance.'''.format(ntuples)))

        if ntuples.dim != grid.ntotal:
            raise ValueError(errfmt('''
            dimension {} of `ntuples` not equal to total number {} of
            grid points.'''.format(ntuples.dim, grid.ntotal)))

        # TODO: make this an `IntervalProd` method (or add to `contains()`)
        if not (np.all(grid.min >= ip_funcset.domain.begin) and
                np.all(grid.max <= ip_funcset.domain.end)):
            raise ValueError(errfmt('''
            `grid` {} not contained in the `domain` {} of `ip_funcset`.
            '''.format(grid, ip_funcset.domain)))

        self._domain = ip_funcset
        self._range = ntuples
        self._grid = grid

    @property
    def domain(self):
        """Return the `domain` attribute."""
        return self._domain

    @property
    def range(self):
        """Return the `range` attribute."""
        return self._range

    @property
    def grid(self):
        """Return the `grid` attribute."""
        return self._grid

    def _call(self, inp):
        """The raw `call` method for out-of-place evaluation.

        Parameters
        ----------
        inp : `FunctionSet.Vector`
            The function to be evaluated. It must accept point
            coordinates in list form (`f(x, y, z)` rather than
            `f(point)`) and return either a NumPy array of the correct
            type (defined by the `Ntuples` instance) or a single value.

        Returns
        -------
        outp : `Ntuples.Vector`
            The function values at the grid points.

        Note
        ----
        The code of this call tries to make use of vectorization of
        the input function, which makes execution much faster and
        memory-saving. If this fails, it falls back to a slow
        loop-based variant.

        Write your function such that every variable occurs -
        otherwise, the values will not be broadcasted to the correct
        size (see example below).

        Avoid using the `numpy.vectorize` function - it is merely a
        convenience function and will not give any speed benefit.

        See also
        --------
        See the `meshgrid` method of `TensorGrid` in `odl.discr.grid`
        or the `numpy.meshgrid` function for an explanation of
        meshgrids.

        Examples
        --------
        Define the grid:

        >>> from odl.discr.grid import TensorGrid
        >>> grid = TensorGrid([1, 2], [3, 4, 5])

        The `ntuples` backend is `Rn`:

        >>> from odl.space.cartesian import Rn
        >>> rn = Rn(grid.ntotal)

        Define a set of functions from the convex hull of the grid
        to the real numbers:

        >>> from odl.space.function import FunctionSet
        >>> from odl.space.set import RealNumbers
        >>> funcset = FunctionSet(grid.convex_hull(), RealNumbers())

        Finally create the operator:

        >>> coll_op = GridCollocation(funcset, grid, rn)
        >>> func_elem = funcset.element(lambda x, y: x - y)
        >>> coll_op(func_elem)
        Rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])

        Use all free variables in functions you supply, otherwise
        the automatic broadcasting will yield a wrong shape:

        >>> func_elem = funcset.element(lambda x, y: 2 * x)
        >>> coll_op(func_elem)
        Traceback (most recent call last):
        ...
        ValueError: `data` shape (2,) not broadcastable to shape (6).

        Do this instead:

        >>> func_elem = funcset.element(lambda x, y: 2 * x + 0 * y)
        >>> coll_op(func_elem)
        Rn(6).element([2.0, 2.0, 2.0, 4.0, 4.0, 4.0])

        This is what happens internally:

        >>> xx, yy = grid.meshgrid()
        >>> vals = 2 * xx
        >>> vals.shape  # Not possible to assign to an Rn(6) vector
        (2, 1)
        """
        # TODO: 'C'-ordering is hard-coded now. Allow 'F' also?
        try:
            mg_tuple = self.grid.meshgrid()
            values = inp.function(*mg_tuple).flatten()
        except TypeError:
            points = self.grid.points()
            values = np.empty(points.shape[0], dtype=self.range.dtype)
            for i, point in enumerate(points):
                values[i] = inp.function(*point)
        return self.range.element(values)


class NearestInterpolation(Operator):

    """Nearest neighbor interpolation as an operator."""

    def __init__(self, ip_funcset, grid, ntuples):
        """Initialize a new `NearestInterpolation` instance.

        Parameters
        ----------
        ip_funcset : `FunctionSet`
            Set of functions, the operator domain. Its `domain` must
            be an `IntervalProduct`.
        grid : `TensorGrid`
            The grid on which to interpolate. Must be contained in
            `ip_funcset.domain`.
        ntuples : `Ntuples`
            Implementation of n-tuples, the operator domain. Its
            dimension must be equal to `grid.ntotal`.
        """
        if not isinstance(ip_funcset, FunctionSet):
            raise TypeError(errfmt('''
            `ip_funcset` {} not an `FunctionSet` instance.
            '''.format(ip_funcset)))

        if not isinstance(ip_funcset.domain, IntervalProd):
            raise TypeError(errfmt('''
            `domain` {} of `ip_funcset` not an `IntervalProd` instance.
            '''.format(ip_funcset.domain)))

        if not isinstance(grid, TensorGrid):
            raise TypeError(errfmt('''
            `grid` {} not a `TensorGrid` instance.
            '''.format(grid)))

        if not isinstance(ntuples, Ntuples):
            raise TypeError(errfmt('''
            `ntuples` {} not an `Ntuples` instance.'''.format(ntuples)))

        if ntuples.dim != grid.ntotal:
            raise ValueError(errfmt('''
            dimension {} of `ntuples` not equal to total number {} of
            grid points.'''.format(ntuples.dim, grid.ntotal)))

        # TODO: make this an `IntervalProd` method (or add to `contains()`)
        if not (np.all(grid.min >= ip_funcset.domain.begin) and
                np.all(grid.max <= ip_funcset.domain.end)):
            raise ValueError(errfmt('''
            `grid` {} not contained in the `domain` {} of `ip_funcset`.
            '''.format(grid, ip_funcset.domain)))

        self._domain = ntuples
        self._range = ip_funcset
        self._grid = grid

    @property
    def domain(self):
        """Return the `domain` attribute."""
        return self._domain

    @property
    def range(self):
        """Return the `range` attribute."""
        return self._range

    @property
    def grid(self):
        """Return the `grid` attribute."""
        return self._grid

    def _call(self, inp):
        """The raw `call` method for out-of-place evaluation.

        Parameters
        ----------
        inp : `Ntuples.Vector`
            The array of numbers to be interpolated

        Returns
        -------
        outp : `IntervalProdFunctionSet.Vector`
            A function (nearest-neighbor) interpolating at a given
            point or array of points.
        """
        def func(x):
            return interpn(points=self.grid.coord_vectors,
                           values=inp.data.reshape(self.grid.shape),
                           method='nearest',
                           xi=x)

        return self.range.element(func)
