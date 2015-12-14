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
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import str, zip

# External imports
from itertools import product
import numpy as np
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

# ODL imports
from odl.discr.grid import TensorGrid
from odl.operator.operator import Operator
from odl.space.base_ntuples import NtuplesBase, FnBase
from odl.space.fspace import FunctionSet, FunctionSpace
from odl.set.domain import IntervalProd
from odl.util.vectorization import is_valid_input_meshgrid


__all__ = ('FunctionSetMapping',
           'GridCollocation', 'NearestInterpolation', 'LinearInterpolation')


class FunctionSetMapping(Operator):

    """Abstract base class for function set discretization mappings."""

    def __init__(self, map_type, fset, grid, dspace, order='C', linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        map_type : {'restriction', 'extension'}
            The type of operator
        fset : `FunctionSet`
            The undiscretized (abstract) set of functions to be
            discretized
        grid :  `TensorGrid`
            The grid on which to evaluate. Must be contained in
            the common domain of the function set.
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object. Its dimension must be equal to the
            total number of grid points.
        order : {'C', 'F'}, optional
            Ordering of the values in the flat data arrays. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        linear : bool
            Create a linear operator if `True`, otherwise a non-linear
            operator.
        """
        map_type_ = str(map_type).lower()
        if map_type_ not in ('restriction', 'extension'):
            raise ValueError('mapping type {!r} not understood.'
                             ''.format(map_type))
        if not isinstance(fset, FunctionSet):
            raise TypeError('function set {!r} is not a `FunctionSet` '
                            'instance.'.format(fset))

        if not isinstance(grid, TensorGrid):
            raise TypeError('grid {!r} is not a `TensorGrid` instance.'
                            ''.format(grid))
        if not isinstance(dspace, NtuplesBase):
            raise TypeError('data space {!r} is not an `NtuplesBase` instance.'
                            ''.format(dspace))

        # TODO: this method is expected to exist, which is the case for
        # interval products. It could be a general optional `Set` method
        if not fset.domain.contains_set(grid):
            raise ValueError('grid {} not contained in the domain {} of the '
                             'function set {}.'.format(grid, fset.domain,
                                                       fset))

        if dspace.size != grid.ntotal:
            raise ValueError('size {} of the data space {} not equal '
                             'to the total number {} of grid points.'
                             ''.format(dspace.size, dspace, grid.ntotal))

        self._order = str(order).upper()
        if self.order not in ('C', 'F'):
            raise ValueError('ordering {!r} not understood.'.format(order))

        dom = fset if map_type_ == 'restriction' else dspace
        ran = dspace if map_type_ == 'restriction' else fset
        Operator.__init__(self, dom, ran, linear=linear)
        self._grid = grid

        if self.is_linear:
            if not isinstance(fset, FunctionSpace):
                raise TypeError('function space {!r} is not a `FunctionSpace` '
                                'instance.'.format(fset))
            if not isinstance(dspace, FnBase):
                raise TypeError('data space {!r} is not an `FnBase` instance.'
                                ''.format(dspace))
            if fset.field != dspace.field:
                raise ValueError('field {} of the function space and field '
                                 '{} of the data space are not equal.'
                                 ''.format(fset.field, dspace.field))

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                isinstance(self, type(other)) and
                self.domain == other.domain and
                self.range == other.range and
                self.grid == other.grid and
                self.order == other.order)

    @property
    def grid(self):
        """The sampling grid."""
        return self._grid

    @property
    def order(self):
        """The axis ordering."""
        return self._order


class GridCollocation(FunctionSetMapping):

    """Function evaluation at grid points.

    Given points :math:`x_1, \dots, x_n \\in \Omega \subset \mathbb{R}^d`,
    the grid collocation operator is defined by

        :math:`\mathcal{C}: \mathcal{X} \\to \mathbb{F}^n`,

        :math:`\mathcal{C}(f) := \\big(f(x_1), \dots, f(x_n)\\big)`,

    where :math:`\mathcal{X}` is any (reasonable) space of functions on
    :math:`\Omega` over the field :math:`\mathbb{F}`.

    The generalization to functions on higher-dimensional sets is
    straightforward.

    This is the default 'restriction' used by all core
    discretization classes.
    """

    def __init__(self, ip_fset, grid, dspace, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        ip_fset : `FunctionSet`
            The undiscretized (abstract) set of functions to be
            discretized. The function domain must be an
            `IntervalProd`.
        grid :  `TensorGrid`
            Grid on which to evaluate. It must be contained in
            the common domain of the function set.
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object. Its size must be equal to the
            total number of grid points.
        order : {'C', 'F'}, optional
            Ordering of the values in the flat data arrays. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        linear = True if isinstance(ip_fset, FunctionSpace) else False
        FunctionSetMapping.__init__(self, 'restriction', ip_fset, grid,
                                    dspace, order, linear=linear)

        # TODO: relax? One needs contains_set() and contains_all()
        if not isinstance(ip_fset.domain, IntervalProd):
            raise TypeError('domain {!r} of the function set is not an '
                            '`IntervalProd` instance.'
                            ''.format(ip_fset.domain))

    def _call(self, func, out=None):
        """Evaluate ``func`` at the grid of this operator..

        Parameters
        ----------
        func : `FunctionSetVector`
            The function to be evaluated
        out : `numpy.ndarray`, optional
            Array to which the values are written

        Returns
        -------
        out : `numpy.ndarray`
            The function values at the grid points. If ``out`` was
            given as argument, it is returned.

        Notes
        -----
        The code of this call tries to make use of vectorization of
        the input function, which makes execution much faster and
        memory-saving. If this fails, it falls back to a slow
        loop-based variant.

        Write your function such that every variable occurs -
        otherwise, the values will not be broadcasted to the correct
        size (see example below).

        See also
        --------
        TensorGrid.meshgrid
        numpy.meshgrid

        Examples
        --------
        >>> from odl import TensorGrid, Rn
        >>> grid = TensorGrid([1, 2], [3, 4, 5], as_midp=True)
        ...
        >>> rn = Rn(grid.ntotal)

        Define a set of functions from the convex hull of the grid
        to the real numbers:

        >>> from odl import FunctionSet, RealNumbers
        >>> funcset = FunctionSet(grid.convex_hull(), RealNumbers())

        Finally create the operator and test it on a function:

        >>> coll_op = GridCollocation(funcset, grid, rn)
        ...
        >>> def func(x):
        ...     return x[0] - x[1]  # properly vectorized
        ...
        >>> func_elem = funcset.element(func)
        >>> coll_op(func_elem)
        Rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])
        >>> coll_op(func)  # Works directly
        Rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])
        >>> out = Rn(6).element()
        >>> coll_op(func, out=out)  # In-place
        Rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])

        Fortran ordering:

        >>> coll_op = GridCollocation(funcset, grid, rn, order='F')
        >>> coll_op(func_elem)
        Rn(6).element([-2.0, -1.0, -3.0, -2.0, -4.0, -3.0])
        """
        mg = self.grid.meshgrid()
        if out is None:
            out = func(mg).ravel(order=self.order)
        else:
            func(mg, out=out.asarray().reshape(self.grid.shape,
                                               order=self.order))
        return out


class NearestInterpolation(FunctionSetMapping):

    """Nearest neighbor interpolation as an `Operator`.

    Given points :math:`x_1, \dots, x_n \\in \mathbb{R}` and values
    :math:`f_1, \dots, f_n \\in \mathbb{F}`, the nearest neighbor
    interpolation at an arbitrary point :math:`x \\in \mathbb{R}`
    is defined by

        :math:`I_{\\bar f}(x) := x_j, \\text{ where } j
        \\text{ is such that } x \\in [x_j, x_{j+1})`

    for :math:`\\bar f := (f_1, \dots, f_n) \\in \mathbb{F}^n`.

    The corresponding nearest neighbor interpolation operator
    is then defined as

        :math:`\mathcal{N}: \mathbb{R}^n \\to \mathcal{X}`,

        :math:`\mathcal{N}(\\bar f) := I_{\\bar f}`,

    where :math:`\mathcal{X}` is any (reasonable) space over the field
    :math:`\mathbb{F}`, of functions on a subset of :math:`\mathbb{R}`.

    The higher-dimensional analog of this operator is simply given
    per component by the above definition.
    """

    def __init__(self, ip_fset, grid, dspace, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        ip_fset : `FunctionSet`
            The undiscretized (abstract) set of functions to be
            discretized. The function domain must be an
            `IntervalProd`.
        grid :  `TensorGrid`
            The grid on which to evaluate. Must be contained in
            the domain of the function set.
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object. Its size must be equal to the
            total number of grid points.
        order : {'C', 'F'}, optional
            Ordering of the values in the flat data arrays. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        linear = True if isinstance(ip_fset, FunctionSpace) else False
        FunctionSetMapping.__init__(self, 'extension', ip_fset, grid, dspace,
                                    order, linear=linear)

        # TODO: relax? One needs contains_set() and contains_all()
        if not isinstance(ip_fset.domain, IntervalProd):
            raise TypeError('domain {!r} of the function set is not an '
                            '`IntervalProd` instance.'
                            ''.format(ip_fset.domain))

    def _call(self, x, out=None):
        """Create an interpolator from grid values ``x``.

        Parameters
        ----------
        x : `NtuplesVector`
            The array of numbers to be interpolated

        Returns
        -------
        out : `function`
            Nearest-neighbor interpolator for the grid of this
            operator.

        See also
        --------
        LinearInterpolation : (bi-/tri-/...)linear interpolation

        Notes
        -----
        **Important:** if called on a point array, the points are
        assumed to be sorted in ascending order in each dimension
        for efficiency reasons.

        Nearest neighbor interpolation is the only scheme which works
        with arbitrary data since it does not involve any arithmetic
        operations on the values.

        Examples
        --------
        >>> from __future__ import unicode_literals, print_function
        >>> from odl import Rectangle, Strings
        >>> rect = Rectangle([0, 0], [1, 1])
        >>> strings = Strings(1)  # 1-char strings

        Initialize the space

        >>> from odl import FunctionSet
        >>> space = FunctionSet(rect, strings)

        The grid is defined by uniform sampling
        (`TensorGrid.as_midp` indicates that the points will
        be cell midpoints instead of corners).

        >>> from odl import uniform_sampling, Ntuples
        >>> grid = uniform_sampling(rect, [4, 2], as_midp=True)
        >>> grid.coord_vectors
        (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))

        >>> dspace = Ntuples(grid.ntotal, dtype='U1')

        Now initialize the operator:

        >>> interp_op = NearestInterpolation(space, grid, dspace,
        ...                                  order='C')

        We test some simple values:

        >>> import numpy as np
        >>> val_arr = np.array([c for c in 'mystring'])
        >>> values = dspace.element(val_arr)
        >>> function = interp_op(values)
        >>> val = function([0.3, 0.6])  # closest to index (1, 1) -> 3
        >>> print(val)
        t
        >>> out = np.empty(2, dtype='U1')
        >>> pts = np.array([[0.3, 0.6],
        ...                 [1.0, 1.0]])
        >>> out = function(pts.T, out=out)  # returns original out
        >>> all(out == ['t', 'g'])
        True
        """
        # TODO: pass reasonable options on to the interpolator
        def nearest(arg, out=None):
            """Interpolating function with vectorization."""
            if is_valid_input_meshgrid(arg, self.grid.ndim):
                # TODO: check if this works for 'F' ordering
                interp = _NearestMeshgridInterpolator(
                    self.grid.coord_vectors,
                    x.data.reshape(self.grid.shape, order=self.order))
            else:
                interp = _NearestPointwiseInterpolator(
                    self.grid.coord_vectors,
                    x.data.reshape(self.grid.shape, order=self.order))
            return interp(arg, out=out)

        return self.range.element(nearest, vectorized=True)


class LinearInterpolation(FunctionSetMapping):

    """Linear interpolation interpolation as a `LinearOperator`."""

    def __init__(self, ip_fspace, grid, dspace, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The undiscretized (abstract) space of functions to be
            discretized. Its field must be the same as that of data
            space. Its `Operator.domain` must be an
            `IntervalProd`.
        grid :  `TensorGrid`
            The grid on which to evaluate. Must be contained in
            the domain of the function set.
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its size must be equal to the
            total number of grid points. Its field must be the same
            as that of the function space.
        order : {'C', 'F'}, optional
            Ordering of the values in the flat data arrays. 'C'
            means the first grid axis varies fastest, the last most
            slowly, 'F' vice versa.
        """
        if not isinstance(ip_fspace, FunctionSpace):
            raise TypeError('function space {!r} is not a `FunctionSpace` '
                            'instance.'.format(ip_fspace))
        if not isinstance(ip_fspace.domain, IntervalProd):
            raise TypeError('function space domain {!r} is not an '
                            '`IntervalProd` instance.'.format(ip_fspace))

        FunctionSetMapping.__init__(self, 'extension', ip_fspace, grid, dspace,
                                    order, linear=True)

    def _call(self, x, out=None):
        """Create an interpolator from grid values ``x``.

        Parameters
        ----------
        x : `FnBaseVector`
            The array of numbers to be interpolated

        Returns
        -------
        out : `function`
            Linear interpolator for the grid of this operator
        """
        def linear(arg, out=None):
            """Interpolating function with vectorization."""
            if is_valid_input_meshgrid(arg, self.grid.ndim):
                # TODO: check if this works for 'F' ordering
                interp = _LinearMeshgridInterpolator(
                    self.grid.coord_vectors,
                    x.data.reshape(self.grid.shape, order=self.order))
            else:
                interp = _LinearPointwiseInterpolator(
                    self.grid.coord_vectors,
                    x.data.reshape(self.grid.shape, order=self.order))
            return interp(arg, out=out)

        return self.range.element(linear, vectorized=True)


class _PointwiseInterpolator(object):

    """Abstract interpolator class for pointwise interpolation.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """

    def __init__(self, coord_vecs, values):
        """Initialize a new instance."""

        # Provide values for some attributes
        self.bounds_error = False
        # TODO: use fill_value depending on discretization
        self.fill_value = None  # extrapolate

        if not hasattr(values, 'ndim'):
            # allow reasonable duck-typed values
            values = np.asarray(values)

        if len(coord_vecs) > values.ndim:
            raise ValueError('There are {} point arrays, but `values` has {} '
                             'dimensions.'.format(len(coord_vecs),
                                                  values.ndim))

        # Cast to floating point was removed here

#        self.fill_value = fill_value
#        if fill_value is not None:
#            fill_value_dtype = np.asarray(fill_value).dtype
#            if (hasattr(values, 'dtype')
#                    and not np.can_cast(fill_value_dtype, values.dtype,
#                                        casting='same_kind')):
#                raise ValueError("fill_value must be either 'None' or "
#                                 "of a type compatible with values")

        for i, p in enumerate(coord_vecs):
            if not np.asarray(p).ndim == 1:
                raise ValueError('The points in dimension {} must be '
                                 '1-dimensional'.format(i))
            if not values.shape[i] == len(p):
                raise ValueError('There are {} points and {} values in '
                                 'dimension {}'.format(len(p),
                                                       values.shape[i], i))

#        for i, p in enumerate(points):
#            if not np.all(np.diff(p) > 0.):
#                raise ValueError("The points in dimension {} must be strictly"
#                                 " ascending".format(i))
        self.grid = tuple([np.asarray(p) for p in coord_vecs])
        self.values = values

    def __call__(self, xi, out=None):
        """Do the interpolation.

        Modified for in-place evaluation support and without method
        choice. Evaluation points are to be given as an array with
        shape (n, dim), where n is the number of points.
        """
        ndim = len(self.grid)
        if xi.ndim != 2:
            raise ValueError('`xi` has {} axes instead of 2.'.format(xi.ndim))

        if xi.shape[0] != ndim:
            raise ValueError('`xi` has axis 1 with length {} instead '
                             'of the grid dimension {}.'.format(xi.shape[0],
                                                                ndim))
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance.'.format(out))
            if out.shape != (xi.shape[0],):
                raise ValueError('Output shape {} not equal to (n,), where '
                                 'n={} is the total number of evaluation '
                                 'points.'.format(out.shape, xi.shape[0]))

        xi = _ndim_coords_from_arrays(xi, ndim=ndim)
        if xi.shape[0] != ndim:
            raise ValueError('The requested sample points xi have dimension '
                             '{}, but this _NearestInterpolator has '
                             'dimension {}.'.format(xi.shape[0], ndim))

#        indices, norm_distances, out_of_bounds = self._find_indices(xi.T)
#        if method == "linear":
#            result = self._evaluate_linear(indices, norm_distances,
#                                           out_of_bounds)
#        elif method == "nearest":
#            result = self._evaluate_nearest(indices, norm_distances,
#                                            out_of_bounds)
#        if not self.bounds_error and self.fill_value is not None:
#            result[out_of_bounds] = self.fill_value

        indices, norm_distances = self._find_indices(xi)
        return self._evaluate(indices, norm_distances, out)

#    def _find_indices(self, xi):
#        # find relevant edges between which xi are situated
#        indices = []
#        # compute distance to lower edge in unity units
#        norm_distances = []
#        # check for out of bounds xi
#        out_of_bounds = np.zeros((xi.shape[1]), dtype=bool)
#        # iterate through dimensions
#        for x, grid in zip(xi, self.grid):
#            i = np.searchsorted(grid, x) - 1
#            i[i < 0] = 0
#            i[i > grid.size - 2] = grid.size - 2
#            indices.append(i)
#            norm_distances.append((x - grid[i]) /
#                                  (grid[i + 1] - grid[i]))
#            if not self.bounds_error:
#                out_of_bounds += x < grid[0]
#                out_of_bounds += x > grid[-1]
#        return indices, norm_distances, out_of_bounds

    def _find_indices(self, xi):
        """Modified version without out-of-bounds check."""
        # find relevant edges between which xi are situated
        indices = []
        # compute distance to lower edge in unity units
        norm_distances = []
        # iterate through dimensions
        for x, grid in zip(xi, self.grid):
            i = np.searchsorted(grid, x) - 1
            i[i < 0] = 0
            i[i > grid.size - 2] = grid.size - 2
            indices.append(i)
            norm_distances.append((x - grid[i]) /
                                  (grid[i + 1] - grid[i]))
        return indices, norm_distances


class _MeshgridInterpolator(_PointwiseInterpolator):

    """Abstract interpolator class for pointwise interpolation.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """

    def __call__(self, xi, out=None):
        """Do the interpolation.

        Modified for in-place evaluation support and without method
        choice. Evaluation points are to be given as a list of arrays
        which can be broadcast against each other.
        """
        if len(xi) != len(self.grid):
            raise ValueError('number of vectors in `xi` is {} instead of {}, '
                             'the grid dimension.'.format(xi.shape[1],
                                                          len(self.grid)))

        if len(xi) == 1:
            ntotal = xi[0].size
        else:
            ntotal = np.prod(np.broadcast(*xi).shape)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance.'.format(out))
            if out.shape != (ntotal,):
                raise ValueError('Output shape {} not equal to (n,), where '
                                 'n={} is the total number of evaluation '
                                 'points.'.format(out.shape, ntotal))

        indices, norm_distances = self._find_indices(xi)
        return self._evaluate(indices, norm_distances, out)


class _NearestPointwiseInterpolator(_PointwiseInterpolator):

    """Nearest neighbor interpolator for point arrays.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate nearest interpolation. Modified for in-place."""
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            # TODO: adapt here for the different cases [a, b), (a, b] or (a, b)
            idx_res.append(np.where(yi <= .5, i, i + 1))
        if out is not None:
            out[:] = self.values[idx_res]
            return out
        else:
            return self.values[idx_res]


class _NearestMeshgridInterpolator(_MeshgridInterpolator,
                                   _NearestPointwiseInterpolator):

    """Nearest neighbor interpolator for point meshgrids.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """


class _LinearPointwiseInterpolator(_PointwiseInterpolator):

    """Linear interpolator for point arrays.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate nearest interpolation. Modified for in-place."""
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        # find relevant values
        # each i and i+1 represents a edge
        edges = product(*[[i, i + 1] for i in indices])
        values = 0.
        for edge_indices in edges:
            weight = 1.
            for ei, i, yi in zip(edge_indices, indices, norm_distances):
                weight *= np.where(ei == i, 1 - yi, yi)
            values += np.asarray(self.values[edge_indices]) * weight[vslice]
        return values


class _LinearMeshgridInterpolator(_MeshgridInterpolator,
                                  _LinearPointwiseInterpolator):

    """Nearest neighbor interpolator for point meshgrids.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
