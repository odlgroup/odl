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
from builtins import str, super, zip

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
            means the first grid axis varies slowest, the last fastest,
            'F' vice versa.
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
            discretized. The function domain must provide a
            ``contains_set`` method as `IntervalProd` does.
        grid :  `TensorGrid`
            Grid on which to evaluate. It must be contained in
            the domain of the function set.
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object. Its size must be equal to the
            total number of grid points.
        order : {'C', 'F'}, optional
            Ordering of the values in the flat data arrays. 'C'
            means the first grid axis varies slowest, the last fastest,
            'F' vice versa.
        """
        linear = isinstance(ip_fset, FunctionSpace)
        FunctionSetMapping.__init__(self, 'restriction', ip_fset, grid,
                                    dspace, order, linear=linear)

    def _call(self, func, out=None):
        """Evaluate ``func`` at the grid of this operator.

        Parameters
        ----------
        func : `FunctionSetVector`
            The function to be evaluated
        out : `NtuplesBaseVector`, optional
            Array to which the values are written. Its shape must be
            ``(N,)``, where N is the total number of grid points. The
            data type must be the same as in the ``dspace`` of this
            mapping.

        Returns
        -------
        out : `NtuplesBaseVector`, optional
            The function values at the grid points. If ``out`` was
            provided, the returned object is a reference to it.

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
        ... # Properly vectorized function
        >>> func_elem = funcset.element(lambda x: x[0] - x[1])
        >>> coll_op(func_elem)
        Rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])
        >>> coll_op(lambda x: x[0] - x[1])  # Works directly
        Rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])
        >>> out = Rn(6).element()
        >>> coll_op(func_elem, out=out)  # In-place
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

    def __init__(self, ip_fset, grid, dspace, order='C', **kwargs):
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
            means the first grid axis varies slowest, the last fastest,
            'F' vice versa.
        variant : {'left', 'right'}, optional
            Behavior variant at midpoint between neighbors

            'left' : favor left neighbor (default)

            'right' : favor right neighbor
        """
        linear = True if isinstance(ip_fset, FunctionSpace) else False
        FunctionSetMapping.__init__(self, 'extension', ip_fset, grid, dspace,
                                    order, linear=linear)

        # TODO: relax? One needs contains_set() and contains_all()
        if not isinstance(ip_fset.domain, IntervalProd):
            raise TypeError('domain {!r} of the function set is not an '
                            '`IntervalProd` instance.'
                            ''.format(ip_fset.domain))

        variant = kwargs.pop('variant', 'left')
        self._variant = str(variant).lower()
        if self._variant not in ('left', 'right'):
            raise ValueError("variant '{}' not understood.".format(variant))

    def _call(self, x, out=None):
        """Create an interpolator from grid values ``x``.

        Parameters
        ----------
        x : `NtuplesVector`
            The array of values to be interpolated
        out : `FunctionSetVector`, optional
            Vector in which to store the interpolator

        Returns
        -------
        out : `FunctionSetVector`
            Nearest-neighbor interpolator for the grid of this
            operator. If ``out`` was provided, the returned object
            is a reference to it.

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
        We test nearest neighbor interpolation with a non-scalar
        data type in 2d:

        >>> from __future__ import print_function
        >>> import numpy as np
        >>> from odl import Rectangle, Strings, FunctionSet
        >>> rect = Rectangle([0, 0], [1, 1])
        >>> strings = Strings(1)  # 1-char strings
        >>> space = FunctionSet(rect, strings)

        The grid is defined by uniform sampling
        (`TensorGrid.as_midp` indicates that the points will
        be cell midpoints instead of corners).

        >>> from odl import uniform_sampling, Ntuples
        >>> grid = uniform_sampling(rect, [4, 2], as_midp=True)
        >>> grid.coord_vectors
        (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))

        >>> dspace = Ntuples(grid.ntotal, dtype='U1')

        Now we initialize the operator and test it with some points:

        >>> interp_op = NearestInterpolation(space, grid, dspace)
        >>> values = np.array([c for c in 'mystring'])
        >>> function = interp_op(values)
        >>> print(function([0.3, 0.6]))  # closest to index (1, 1) -> 3
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
                interp = _NearestMeshgridInterpolator(
                    self.grid.coord_vectors,
                    x.data.reshape(self.grid.shape, order=self.order),
                    variant=self._variant)
            else:
                interp = _NearestPointwiseInterpolator(
                    self.grid.coord_vectors,
                    x.data.reshape(self.grid.shape, order=self.order),
                    variant=self._variant)

            return interp(arg, out=out)

        return self.range.element(nearest, vectorized=True)


class LinearInterpolation(FunctionSetMapping):

    """Linear interpolation interpolation as an `Operator`."""

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
            means the first grid axis varies slowest, the last fastest,
            'F' vice versa.
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
            The array of values to be interpolated
        out : `FunctionSpaceVector`, optional
            Vector in which to store the interpolator

        Returns
        -------
        out : `FunctionSpaceVector`
            Linear interpolator for the grid of this operator. If
            ``out`` was provided, the returned object is a reference
            to it.
        """
        def linear(arg, out=None):
            """Interpolating function with vectorization."""
            if is_valid_input_meshgrid(arg, self.grid.ndim):
                # TODO: check if this works for 'F' ordering
                interp = _LinearMeshgridInterpolator(
                    self.grid.coord_vectors,
                    x.data.reshape(self.grid.shape, order=self.order))
            else:
                # TODO: check if points are ordered. If not, use
                # LinearNDInterpolator
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

        values = np.asarray(values)

        if len(coord_vecs) > values.ndim:
            raise ValueError('There are {} point arrays, but `values` has {} '
                             'dimensions.'.format(len(coord_vecs),
                                                  values.ndim))

        # Cast to floating point was removed here

        for i, p in enumerate(coord_vecs):
            if not np.asarray(p).ndim == 1:
                raise ValueError('The points in dimension {} must be '
                                 '1-dimensional'.format(i))
            if values.shape[i] != len(p):
                raise ValueError('There are {} points and {} values in '
                                 'dimension {}'.format(len(p),
                                                       values.shape[i], i))

        self.grid = tuple([np.asarray(p) for p in coord_vecs])
        self.values = values

    def __call__(self, x, out=None):
        """Do the interpolation.

        Modified for in-place evaluation support and without method
        choice. Evaluation points are to be given as an array with
        shape (n, dim), where n is the number of points.
        """
        ndim = len(self.grid)
        if ndim == 1 and x.ndim == 1:
            x = x[None, :]

        if x.ndim != 2:
            raise ValueError('x has {} axes instead of 2.'.format(x.ndim))

        if x.shape[0] != ndim:
            raise ValueError('x has axis 1 with length {} instead '
                             'of the grid dimension {}.'
                             ''.format(x.shape[0], ndim))
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance.'.format(out))
            if out.shape != (x.shape[1],):
                raise ValueError('Output shape is {}, expected ({n},) since '
                                 'there are {n} evaluation points.'
                                 ''.format(out.shape, n=x.shape[1]))

        x = _ndim_coords_from_arrays(x, ndim=ndim)
        if x.shape[0] != ndim:
            raise ValueError('The requested sample points x have dimension '
                             '{}, but this _NearestInterpolator has '
                             'dimension {}.'.format(x.shape[0], ndim))

        indices, norm_distances = self._find_indices(x)
        return self._evaluate(indices, norm_distances, out)

    def _find_indices(self, x):
        """Find indices and distances of the given nodes."""

        # find relevant edges between which xi are situated
        index_vecs = []
        # compute distance to lower edge in unity units
        norm_distances = []

        # iterate through dimensions
        for xi, cvec in zip(x, self.grid):
            idcs = np.searchsorted(cvec, xi) - 1

            idcs[idcs < 0] = 0
            idcs[idcs > cvec.size - 2] = cvec.size - 2
            index_vecs.append(idcs)

            norm_distances.append((xi - cvec[idcs]) /
                                  (cvec[idcs + 1] - cvec[idcs]))

        return index_vecs, norm_distances


class _MeshgridInterpolator(_PointwiseInterpolator):

    """Abstract interpolator class for pointwise interpolation.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """

    def __call__(self, x, out=None):
        """Do the interpolation.

        Modified for in-place evaluation support and without method
        choice. Evaluation points are to be given as a list of arrays
        which can be broadcast against each other.
        """
        if len(x) != len(self.grid):
            raise ValueError('number of vectors in x is {} instead of {}, '
                             'the grid dimension.'
                             ''.format(len(x), len(self.grid)))

        if len(x) == 1:
            out_shape = x[0].shape
        else:
            out_shape = np.broadcast(*x).shape
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance.'.format(out))
            if out.shape != out_shape:
                raise ValueError('Output shape {} not equal to expected '
                                 'shape {}.'.format(out.shape, out_shape))

        indices, norm_distances = self._find_indices(x)
        return self._evaluate(indices, norm_distances, out)


class _NearestPointwiseInterpolator(_PointwiseInterpolator):

    """Nearest neighbor interpolator for point arrays.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.
    """

    def __init__(self, coord_vecs, values, variant):
        super().__init__(coord_vecs, values)
        self.variant = variant

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate nearest interpolation.

        Modified for in-place and added variants."""
        idx_res = []
        for i, yi in zip(indices, norm_distances):
            if self.variant == 'left':
                idx_res.append(np.where(yi <= .5, i, i + 1))
            else:
                idx_res.append(np.where(yi < .5, i, i + 1))
        if out is not None:
            out[:] = self.values[idx_res]
            return out
        else:
            return self.values[idx_res]


class _NearestMeshgridInterpolator(_MeshgridInterpolator,
                                   _NearestPointwiseInterpolator):

    """Nearest neighbor interpolator for point meshgrids."""


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

    """Linear interpolator for point meshgrids."""


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
