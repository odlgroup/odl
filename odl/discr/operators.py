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
from scipy.interpolate import interpn, RegularGridInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

# ODL imports
from odl.discr.grid import TensorGrid
from odl.operator.operator import Operator, LinearOperator
from odl.space.cartesian import Ntuples, Rn, Cn
from odl.space.function import FunctionSet, FunctionSpace
from odl.space.domain import IntervalProd
from odl.utility.utility import errfmt


class RawGridCollocation(Operator):

    """Function evaluation at grid points.

    This is the raw `Operator` version of the default 'restriction'
    used by all core discretization classes.
    """

    def __init__(self, ip_fset, grid, ntuples, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        ip_fset : `FunctionSet`
            Set of functions, the operator range. Its `domain` must
            be an `IntervalProd`.
        grid : `TensorGrid`
            The grid on which to evaluate. Must be contained in
            `ip_fset.domain`.
        ntuples : `Ntuples`
            Implementation of n-tuples, the operator domain. Its
            dimension must be equal to `grid.ntotal`.
        order : 'C' or 'F', optional
            Ordering of the values in the flat `ntuples` array. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        if not isinstance(ip_fset, FunctionSet):
            raise TypeError(errfmt('''
            `ip_fset` {} not an `FunctionSet` instance.
            '''.format(ip_fset)))

        if not isinstance(ip_fset.domain, IntervalProd):
            raise TypeError(errfmt('''
            `domain` {} of `ip_fset` not an `IntervalProd` instance.
            '''.format(ip_fset.domain)))

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
        if not (np.all(grid.min >= ip_fset.domain.begin) and
                np.all(grid.max <= ip_fset.domain.end)):
            raise ValueError(errfmt('''
            `grid` {} not contained in the `domain` {} of `ip_fset`.
            '''.format(grid, ip_fset.domain)))

        if order not in ('C', 'F'):
            raise ValueError('`order` {} not understood.'.format(order))

        self._domain = ip_fset
        self._range = ntuples
        self._grid = grid
        self._order = order

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

    @property
    def order(self):
        """The axis ordering."""
        return self._order

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

        >>> coll_op = RawGridCollocation(funcset, grid, rn)
        >>> func_elem = funcset.element(lambda x, y: x - y)
        >>> coll_op(func_elem)
        Rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])

        Or, if we want Fortran ordering:

        >>> coll_op = RawGridCollocation(funcset, grid, rn, order='F')
        >>> coll_op(func_elem)
        Rn(6).element([-2.0, -1.0, -3.0, -2.0, -4.0, -3.0])

        Use all free variables in functions you supply, otherwise
        the automatic broadcasting will yield a wrong shape:

        >>> func_elem = funcset.element(lambda x, y: 2 * x)
        >>> coll_op(func_elem)
        Traceback (most recent call last):
        ...
        ValueError: `inp` shape (2,) not broadcastable to shape (6,).

        Do this instead:

        >>> func_elem = funcset.element(lambda x, y: 2 * x + 0 * y)
        >>> coll_op(func_elem)
        Rn(6).element([2.0, 4.0, 2.0, 4.0, 2.0, 4.0])

        This is what happens internally:

        >>> xx, yy = grid.meshgrid()
        >>> vals = 2 * xx
        >>> vals.shape  # Not possible to assign to an Rn(6) vector
        (2, 1)
        """
        try:
            mg_tuple = self.grid.meshgrid()
            values = inp(*mg_tuple).flatten(order=self.order)
        except TypeError:
            points = self.grid.points(order=self.order)
            values = np.empty(points.shape[0], dtype=self.range.dtype)
            for i, point in enumerate(points):
                values[i] = inp(*point)
        return self.range.element(values)


class GridCollocation(RawGridCollocation, LinearOperator):

    """Function evaluation at grid points.

    This is the `LinearOperator` version of the default 'restriction'
    used by all core discretization classes.
    """

    def __init__(self, ip_fspace, grid, ntuples, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        ip_fspace : `FunctionSet`
            Space of functions, the operator range. Its `domain` must
            be an `IntervalProd`.
        grid : `TensorGrid`
            The grid on which to evaluate. Must be contained in
            `ip_fspace.domain`.
        ntuples : `Rn` or `Cn`
            Implementation of n-tuples, the operator domain. Its
            dimension must be equal to `grid.ntotal`.
        order : 'C' or 'F', optional
            Ordering of the values in the flat `ntuples` array. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        if not isinstance(ip_fspace, FunctionSpace):
            raise TypeError(errfmt('''
            `ip_fspace` {} not an instance of `FunctionSpace`.
            '''.format(ip_fspace)))

        if not isinstance(ip_fspace.domain, IntervalProd):
            raise TypeError(errfmt('''
            `domain` {} of `ip_fspace` not an instance of
            `IntervalProd`.'''.format(ip_fspace.domain)))

        if not isinstance(grid, TensorGrid):
            raise TypeError(errfmt('''
            `grid` {} not a `TensorGrid` instance.
            '''.format(grid)))

        if not isinstance(ntuples, (Rn, Cn)):
            raise TypeError(errfmt('''
            `ntuples` {} not an instance of `Rn` or `Cn`.
            '''.format(ntuples)))

        if ntuples.dim != grid.ntotal:
            raise ValueError(errfmt('''
            dimension {} of `ntuples` not equal to total number {} of
            grid points.'''.format(ntuples.dim, grid.ntotal)))

        if ip_fspace.field != ntuples.field:
            raise ValueError(errfmt('''
            `field` {} of `ip_fspace` not equal to `field` {}
            of `ntuples`.'''.format(ip_fspace.field, ntuples.field)))

        super().__init__(ip_fspace, grid, ntuples, order)


class RawNearestInterpolation(Operator):

    """Nearest neighbor interpolation as a raw `Operator`."""

    def __init__(self, ip_fset, grid, ntuples, order='C'):
        """Initialize a new `NearestInterpolation` instance.

        Parameters
        ----------
        ip_fset : `FunctionSet`
            Set of functions, the operator domain. Its `domain` must
            be an `IntervalProd`.
        grid : `TensorGrid`
            The grid on which to interpolate. Must be contained in
            `ip_fset.domain`.
        ntuples : `Ntuples`
            Implementation of n-tuples, the operator domain. Its
            dimension must be equal to `grid.ntotal`.
        order : 'C' or 'F', optional
            Ordering of the values in the flat `ntuples` array. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        if not isinstance(ip_fset, FunctionSet):
            raise TypeError(errfmt('''
            `ip_fset` {} not an `FunctionSet` instance.
            '''.format(ip_fset)))

        if not isinstance(ip_fset.domain, IntervalProd):
            raise TypeError(errfmt('''
            `domain` {} of `ip_fset` not an `IntervalProd` instance.
            '''.format(ip_fset.domain)))

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
        if not (np.all(grid.min >= ip_fset.domain.begin) and
                np.all(grid.max <= ip_fset.domain.end)):
            raise ValueError(errfmt('''
            `grid` {} not contained in the `domain` {} of `ip_fset`.
            '''.format(grid, ip_fset.domain)))

        if order not in ('C', 'F'):
            raise ValueError('`order` {} not understood.'.format(order))

        self._domain = ntuples
        self._range = ip_fset
        self._grid = grid
        self._order = order

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

    @property
    def order(self):
        """The axis ordering."""
        return self._order

    def _call(self, inp):
        """The raw `call` method for out-of-place evaluation.

        Parameters
        ----------
        inp : `Ntuples.Vector`
            The array of numbers to be interpolated

        Returns
        -------
        outp : `FunctionSet.Vector`
            A function (nearest-neighbor) interpolating at a given
            point or array of points.

        Examples
        --------
        Let's define the complex function space :math:`L^2` on a
        rectangle:

        >>> from odl.space.domain import Rectangle
        >>> from odl.space.default import L2
        >>> from odl.space.set import ComplexNumbers
        >>> from odl.space.cartesian import Cn

        >>> rect = Rectangle([0, 0], [1, 1])
        >>> space = L2(rect, field=ComplexNumbers())

        The grid is defined by uniform sampling (`as_midp` indicates
        that the points will be cell midpoints instead of corners).

        >>> grid = rect.uniform_sampling([4, 2], as_midp=True)
        >>> grid.coord_vectors
        (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))
        >>> ntuples = Cn(grid.ntotal)

        Now initialize the operator:

        >>> interp_op = RawNearestInterpolation(space, grid, ntuples,
        ...                                     order='C')

        We test some simple values:

        >>> import numpy as np
        >>> val_arr = np.arange(8) + 1j * np.arange(1, 9)
        >>> values = ntuples.element(val_arr)
        >>> function = interp_op(values)
        >>> function(0.3, 0.6)  # closest to index (1, 1) -> 5

        """
        def func(*x):
            x = np.atleast_1d(x).squeeze()
            interp = _NearestInterpolator(
                self.grid.coord_vectors, inp.data.reshape(self.grid.shape,
                                                          order=self.order))
            values = interp(x)
            return values[0] if values.shape == (1,) else values

        return self.range.element(func)


class NearestInterpolation(RawNearestInterpolation, LinearOperator):

    """Nearest neighbor interpolation as a `LinearOperator`."""

    def __init__(self, ip_fspace, grid, ntuples, order='C'):
        """Initialize a new `NearestInterpolation` instance.

        Parameters
        ----------
        ip_fspace : `FunctionSpace`
            Space of functions, the operator domain. Its `domain` must
            be an `IntervalProd`.
        grid : `TensorGrid`
            The grid on which to interpolate. Must be contained in
            `ip_fset.domain`.
        ntuples : `Rn` or `Cn`
            Implementation of n-tuples, the operator domain. Its
            dimension must be equal to `grid.ntotal`.
        order : 'C' or 'F', optional
            Ordering of the values in the flat `ntuples` array. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        if not isinstance(ip_fspace, FunctionSpace):
            raise TypeError(errfmt('''
            `ip_fspace` {} not an instance of `FunctionSpace`.
            '''.format(ip_fspace)))

        if not isinstance(ip_fspace.domain, IntervalProd):
            raise TypeError(errfmt('''
            `domain` {} of `ip_fspace` not an instance of
            `IntervalProd`.'''.format(ip_fspace.domain)))

        if not isinstance(grid, TensorGrid):
            raise TypeError(errfmt('''
            `grid` {} not a `TensorGrid` instance.
            '''.format(grid)))

        if not isinstance(ntuples, (Rn, Cn)):
            raise TypeError(errfmt('''
            `ntuples` {} not an instance of `Rn` or `Cn`.
            '''.format(ntuples)))

        if ntuples.dim != grid.ntotal:
            raise ValueError(errfmt('''
            dimension {} of `ntuples` not equal to total number {} of
            grid points.'''.format(ntuples.dim, grid.ntotal)))

        if ip_fspace.field != ntuples.field:
            raise ValueError(errfmt('''
            `field` {} of `ip_fspace` not equal to `field` {}
            of `ntuples`.'''.format(ip_fspace.field, ntuples.field)))

        super().__init__(ip_fspace, grid, ntuples)


class LinearInterpolation(LinearOperator):

    """Linear interpolation interpolation as a `LinearOperator`."""

    def __init__(self, ip_fspace, grid, ntuples, order='C'):
        """Initialize a new `NearestInterpolation` instance.

        Parameters
        ----------
        ip_fspace : `FunctionSpace`
            Space of functions, the operator domain. Its `domain` must
            be an `IntervalProd`.
        grid : `TensorGrid`
            The grid on which to interpolate. Must be contained in
            `ip_fset.domain`.
        ntuples : `Rn` or `Cn`
            Implementation of n-tuples, the operator domain. Its
            dimension must be equal to `grid.ntotal`.
        order : 'C' or 'F', optional
            Ordering of the values in the flat `ntuples` array. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        if not isinstance(ip_fspace, FunctionSpace):
            raise TypeError(errfmt('''
            `ip_fspace` {} not an instance of `FunctionSpace`.
            '''.format(ip_fspace)))

        if not isinstance(ip_fspace.domain, IntervalProd):
            raise TypeError(errfmt('''
            `domain` {} of `ip_fspace` not an instance of
            `IntervalProd`.'''.format(ip_fspace.domain)))

        if not isinstance(grid, TensorGrid):
            raise TypeError(errfmt('''
            `grid` {} not a `TensorGrid` instance.
            '''.format(grid)))

        if not isinstance(ntuples, (Rn, Cn)):
            raise TypeError(errfmt('''
            `ntuples` {} not an instance of `Rn` or `Cn`.
            '''.format(ntuples)))

        if ntuples.dim != grid.ntotal:
            raise ValueError(errfmt('''
            dimension {} of `ntuples` not equal to total number {} of
            grid points.'''.format(ntuples.dim, grid.ntotal)))

        if ip_fspace.field != ntuples.field:
            raise ValueError(errfmt('''
            `field` {} of `ip_fspace` not equal to `field` {}
            of `ntuples`.'''.format(ip_fspace.field, ntuples.field)))

        # TODO: make this an `IntervalProd` method (or add to `contains()`)
        if not (np.all(grid.min >= ip_fspace.domain.begin) and
                np.all(grid.max <= ip_fspace.domain.end)):
            raise ValueError(errfmt('''
            `grid` {} not contained in the `domain` {} of `ip_fspace`.
            '''.format(grid, ip_fspace.domain)))

        if order not in ('C', 'F'):
            raise ValueError('`order` {} not understood.'.format(order))

        self._domain = ntuples
        self._range = ip_fspace
        self._grid = grid
        self._order = order

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

    @property
    def order(self):
        """The axis ordering."""
        return self._order

    def _call(self, inp):
        """The raw `call` method for out-of-place evaluation.

        Parameters
        ----------
        inp : `Ntuples.Vector`
            The array of numbers to be interpolated

        Returns
        -------
        outp : `FunctionSet.Vector`
            A function (nearest-neighbor) interpolating at a given
            point or array of points.

        Examples
        --------
        Let's define the complex function space :math:`L^2` on a
        rectangle:

        >>> from odl.space.domain import Rectangle
        >>> from odl.space.default import L2
        >>> from odl.space.set import ComplexNumbers
        >>> from odl.space.cartesian import Cn

        >>> rect = Rectangle([0, 0], [1, 1])
        >>> space = L2(rect, field=ComplexNumbers())

        The grid is defined by uniform sampling (`as_midp` indicates
        that the points will be cell midpoints instead of corners).

        >>> grid = rect.uniform_sampling([4, 2], as_midp=True)
        >>> grid.coord_vectors
        (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))
        >>> ntuples = Cn(grid.ntotal)

        Now initialize the operator:

        TODO: implement an example!

        """
        def func(*x):
            x = np.atleast_1d(x).squeeze()
            values = interpn(points=self.grid.coord_vectors,
                             values=inp.data.reshape(self.grid.shape,
                                                     order=self.order),
                             method='nearest',
                             xi=x,
                             fill_value=None)  # Allow points outside
            return values[0] if values.shape == (1,) else values

        return self.range.element(func)


class _NearestInterpolator(RegularGridInterpolator):

    """Own version of NumPy's grid interpolator class.

    We want to support non-numerical values for nearest neighbor
    interpolation and in-place evaluation.
    """

    def __init__(self, coord_vecs, values):
        """Initialize a new instance."""

        # Provide values for some attributes
        self.method = 'nearest'
        self.bounds_error = False
        self.fill_value = None

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(coord_vecs) > values.ndim:
            raise ValueError('There are {} point arrays, but `values` has {} '
                             'dimensions.'.format(len(coord_vecs),
                                                  values.ndim))

        # Cast to floating point was removed here

        for i, p in enumerate(coord_vecs):
            if not np.all(np.diff(p) > 0.):
                raise ValueError('The points in dimension {} must be strictly '
                                 'ascending'.format(i))
            if not np.asarray(p).ndim == 1:
                raise ValueError('The points in dimension {} must be '
                                 '1-dimensional'.format(i))
            if not values.shape[i] == len(p):
                raise ValueError('There are {} points and {} values in '
                                 'dimension {}'.format(len(p),
                                                       values.shape[i], i))
        self.grid = tuple([np.asarray(p) for p in coord_vecs])
        self.values = values

    def __call__(self, xi, outp=None):
        """Do the interpolation. Modified for in-place support."""
        ntotal = np.prod([len(v) for v in self.grid])
        if outp is not None:
            if not isinstance(outp, np.ndarray):
                raise TypeError('`outp` {!r} not a `numpy.ndarray` '
                                'instance.'.format(outp))
            if outp.shape != (ntotal,):
                raise ValueError('Output shape {} not equal to (n,), where '
                                 'n={} is the total number of grid '
                                 'points.'.format(outp.shape, ntotal))

        ndim = len(self.grid)
        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[-1] != ndim:
            raise ValueError('The requested sample points xi have dimension '
                             '{}, but this _NearestInterpolator has '
                             'dimension {}.'.format(xi.shape[-1], ndim))

        xi_shape = xi.shape
        xi = xi.reshape(-1, xi_shape[-1])
        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)

        outp = self._evaluate_nearest(indices, norm_distances, out_of_bounds,
                                      outp)

        return outp  # Do not reshape

    def _evaluate_nearest(self, indices, norm_distances, out_of_bounds,
                          outp=None):
            """Evaluate nearest interpolation. Modified for in-place."""
            idx_res = []
            for i, yi in zip(indices, norm_distances):
                idx_res.append(np.where(yi <= .5, i, i + 1))
            if outp is not None:
                outp[:] = self.values[idx_res]
                return outp
            else:
                return self.values[idx_res]
