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

"""Mappings between abstract (continuous) and discrete sets.

Includes grid evaluation (collocation) and various interpolation
operators.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import str, super, zip

# External imports
from itertools import product
import numpy as np

# ODL imports
from odl.discr.grid import TensorGrid
from odl.operator.operator import Operator
from odl.space.base_ntuples import NtuplesBase, FnBase
from odl.space.fspace import FunctionSet, FunctionSpace
from odl.set.domain import IntervalProd
from odl.util.vectorization import (
    is_valid_input_meshgrid, out_shape_from_array, out_shape_from_meshgrid)


__all__ = ('FunctionSetMapping',
           'GridCollocation', 'NearestInterpolation', 'LinearInterpolation',
           'PerAxisInterpolation')

_SUPPORTED_INTERP_SCHEMES = ['nearest', 'linear']


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
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points.
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

        if dspace.size != grid.size:
            raise ValueError('size {} of the data space {} not equal '
                             'to the total number {} of grid points.'
                             ''.format(dspace.size, dspace, grid.size))

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
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points.
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
        odl.discr.grid.TensorGrid.meshgrid
        numpy.meshgrid

        Examples
        --------
        >>> from odl import TensorGrid, Rn
        >>> grid = TensorGrid([1, 2], [3, 4, 5], as_midp=True)
        ...
        >>> rn = Rn(grid.size)

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
        try:
            mesh = self.grid.meshgrid()
            if out is None:
                out = func(mesh).ravel(order=self.order)
            else:
                func(mesh, out=out.asarray().reshape(self.grid.shape,
                                                     order=self.order))
        except (ValueError, TypeError) as err:
            if str(err.args[0]).startswith('output contains points outside'):
                raise err
            points = self.grid.points()
            if out is None:
                out = func(points)
            else:
                func(points, out=out.asarray())
        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n  {!r},\n  {!r},\n  {!r}'.format(self.domain,
                                                        self.grid,
                                                        self.range)
        if self.order == 'F':
            inner_str += ",\n  order='F'"

        return '{}({})'.format(self.__class__.__name__, inner_str)


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
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points.
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

        >>> import numpy as np
        >>> from odl import Rectangle, Strings, FunctionSet
        >>> rect = Rectangle([0, 0], [1, 1])
        >>> strings = Strings(1)  # 1-char strings
        >>> space = FunctionSet(rect, strings)

        The grid is defined by uniform sampling
        (`TensorGrid.as_midp` indicates that the points will
        be cell midpoints instead of corners).

        >>> from odl import uniform_sampling_fromintv, Ntuples
        >>> grid = uniform_sampling_fromintv(rect, [4, 2], as_midp=True)
        >>> grid.coord_vectors
        (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))

        >>> dspace = Ntuples(grid.size, dtype='U1')

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
                input_type = 'meshgrid'
            else:
                input_type = 'array'

            interpolator = _NearestInterpolator(
                self.grid.coord_vectors,
                x.data.reshape(self.grid.shape, order=self.order),
                variant=self._variant,
                input_type=input_type)

            return interpolator(arg, out=out)

        return self.range.element(nearest, vectorized=True)

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n  {!r},\n  {!r},\n  {!r}'.format(self.range,
                                                        self.grid,
                                                        self.domain)
        if self.order == 'F':
            inner_str += ",\n  order='F'"
        if self._variant == 'right':
            inner_str += ",\n  variant='right'"

        return '{}({})'.format(self.__class__.__name__, inner_str)


class LinearInterpolation(FunctionSetMapping):

    """Linear interpolation interpolation as an `Operator`."""

    def __init__(self, ip_fspace, grid, dspace, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The undiscretized (abstract) space of functions to be
            discretized. Its field must be the same as that of data
            space. Its `FunctionSet.domain` must be an
            `IntervalProd`.
        grid :  `TensorGrid`
            The grid on which to evaluate. Must be contained in
            the domain of the function set.
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points, and its `FnBase.field`
            must be the same as that of the function space.
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
        # TODO: pass reasonable options on to the interpolator
        def linear(arg, out=None):
            """Interpolating function with vectorization."""
            if is_valid_input_meshgrid(arg, self.grid.ndim):
                input_type = 'meshgrid'
            else:
                input_type = 'array'

            interpolator = _LinearInterpolator(
                self.grid.coord_vectors,
                x.data.reshape(self.grid.shape, order=self.order),
                input_type=input_type)

            return interpolator(arg, out=out)

        return self.range.element(linear, vectorized=True)

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n  {!r},\n  {!r},\n  {!r}'.format(self.range,
                                                        self.grid,
                                                        self.domain)
        if self.order == 'F':
            inner_str += ",\n  order='F'"

        return '{}({})'.format(self.__class__.__name__, inner_str)


class PerAxisInterpolation(FunctionSetMapping):

    """Interpolation scheme set for each axis individually."""

    def __init__(self, ip_fspace, grid, dspace, schemes, order='C',
                 nn_variants=None):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The undiscretized (abstract) space of functions to be
            discretized. Its field must be the same as that of data
            space. Its `FunctionSet.domain` must be an
            `IntervalProd`.
        grid :  `TensorGrid`
            The grid on which to evaluate. Must be contained in
            the domain of the function set.
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points, and its `FnBase.field`
            must be the same as that of the function space.
        schemes : `str` or sequence of `str`
            Indicates which interpolation scheme to use for which axis.
            A single string is interpreted as a global scheme for all
            axes.
        order : {'C', 'F'}, optional
            Ordering of the values in the flat data arrays. 'C'
            means the first grid axis varies slowest, the last fastest,
            'F' vice versa.
        nn_variants : `str` or sequence of `str`, optional
            Which variant ('left' or 'right') to use in nearest neighbor
            interpolation for which axis. A single string is interpreted
            as a global variant for all axes.
            This option has no effect for schemes other than nearest
            neighbor.
        """
        if not isinstance(ip_fspace, FunctionSpace):
            raise TypeError('function space {!r} is not a `FunctionSpace` '
                            'instance.'.format(ip_fspace))
        if not isinstance(ip_fspace.domain, IntervalProd):
            raise TypeError('function space domain {!r} is not an '
                            '`IntervalProd` instance.'.format(ip_fspace))

        FunctionSetMapping.__init__(self, 'extension', ip_fspace, grid, dspace,
                                    order, linear=True)

        try:
            schemes_ = str(schemes + '').lower()  # pythonic string check
            schemes_ = [schemes_] * self.grid.ndim
        except TypeError:
            schemes_ = [str(scm).lower() if scm is not None else None
                        for scm in schemes]

        if nn_variants is None:
            variants_ = ['left' if scm == 'nearest' else None
                         for scm in schemes]
        else:
            try:
                variants_ = str(nn_variants + '').lower()  # pythonic str check
                variants_ = [variants_ if scm == 'nearest' else None
                             for scm in schemes]
            except TypeError:
                variants_ = [str(var).lower() if var is not None else None
                             for var in nn_variants]

        for i, (scm, var) in enumerate(zip(schemes_, variants_)):
            if scm not in _SUPPORTED_INTERP_SCHEMES:
                raise ValueError("Interpolation scheme '{}' at index {} not "
                                 "understood.".format(scm, i))
            if scm == 'nearest' and var not in ('left', 'right'):
                raise ValueError("Nearest neighbor variant '{}' at index {} "
                                 "not understood.".format(var, i))
            elif scm != 'nearest' and var is not None:
                raise ValueError('Option nn_variants used in axis {} with '
                                 'scheme {!r}.'.format(i, scm))

        self._schemes = schemes_
        self._nn_variants = variants_

    @property
    def schemes(self):
        """List of interpolation schemes, one for each axis."""
        return self._schemes

    @property
    def nn_variants(self):
        """List of nearest neighbor variants, one for each axis."""
        return self._nn_variants

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
            Per-axis interpolator for the grid of this operator. If
            ``out`` was provided, the returned object is a reference
            to it.
        """
        def per_axis_interp(arg, out=None):
            """Interpolating function with vectorization."""
            if is_valid_input_meshgrid(arg, self.grid.ndim):
                input_type = 'meshgrid'
            else:
                input_type = 'array'

            interpolator = _PerAxisInterpolator(
                self.grid.coord_vectors,
                x.data.reshape(self.grid.shape, order=self.order),
                schemes=self.schemes, nn_variants=self.nn_variants,
                input_type=input_type)

            return interpolator(arg, out=out)

        return self.range.element(per_axis_interp, vectorized=True)

    def __repr__(self):
        """Return ``repr(self)``."""
        if all(scm == self.schemes[0] for scm in self.schemes):
            schemes = self.schemes[0]
        else:
            schemes = self.schemes

        inner_str = '\n  {!r},\n  {!r},\n  {!r},\n  {!r}'.format(
            self.range, self.grid, self.domain, schemes)
        if self.order == 'F':
            inner_str += ",\n  order='F'"

        if all(var == self.nn_variants[0] for var in self.nn_variants):
            variants = self.nn_variants[0]
        else:
            variants = self.nn_variants

        if variants is not None:
            inner_str += ',\n  nn_variants={}'.format(variants)

        return '{}({})'.format(self.__class__.__name__, inner_str)


class _Interpolator(object):

    """Abstract interpolator class.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.

    The init method does not convert to floating point to
    support arbitrary data type for nearest neighbor interpolation.

    Subclasses need to override ``_evaluate`` for concrete
    implementations.
    """

    def __init__(self, coord_vecs, values, input_type):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`
            Coordinate vectors defining the interpolation grid
        values : array-like
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        values = np.asarray(values)
        typ_ = str(input_type).lower()
        if typ_ not in ('array', 'meshgrid'):
            raise ValueError("Type '{}' not understood.".format(input_type))

        if len(coord_vecs) > values.ndim:
            raise ValueError('There are {} point arrays, but `values` has {} '
                             'dimensions.'.format(len(coord_vecs),
                                                  values.ndim))
        for i, p in enumerate(coord_vecs):
            if not np.asarray(p).ndim == 1:
                raise ValueError('The points in dimension {} must be '
                                 '1-dimensional'.format(i))
            if values.shape[i] != len(p):
                raise ValueError('There are {} points and {} values in '
                                 'dimension {}'.format(len(p),
                                                       values.shape[i], i))

        self.coord_vecs = tuple(np.asarray(p) for p in coord_vecs)
        self.values = values
        self.input_type = input_type

    def __call__(self, x, out=None):
        """Do the interpolation.

        Parameters
        ----------
        x : meshgrid or `numpy.ndarray`
            Evaluation points of the interpolator
        out : `numpy.ndarray`, optional
            Array to which the results are written. Needs to have
            correct shape according to input ``x``.

        Returns
        -------
        out : `numpy.ndarray`
            Interpolated values. If ``out`` was given, the returned
            object is a reference to it.
        """
        ndim = len(self.coord_vecs)
        if self.input_type == 'array':
            # Make a (1, n) array from one with shape (n,)
            x = x.reshape([ndim, -1])
            out_shape = out_shape_from_array(x)
        else:
            if len(x) != ndim:
                raise ValueError('number of vectors in x is {} instead of '
                                 'the grid dimension {}.'
                                 ''.format(len(x), ndim))
            out_shape = out_shape_from_meshgrid(x)

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance.'.format(out))
            if out.shape != out_shape:
                raise ValueError('Output shape {} not equal to expected '
                                 'shape {}.'.format(out.shape, out_shape))

        indices, norm_distances = self._find_indices(x)
        return self._evaluate(indices, norm_distances, out)

    def _find_indices(self, x):
        """Find indices and distances of the given nodes."""
        # find relevant edges between which xi are situated
        index_vecs = []
        # compute distance to lower edge in unity units
        norm_distances = []

        # iterate through dimensions
        for xi, cvec in zip(x, self.coord_vecs):
            idcs = np.searchsorted(cvec, xi) - 1

            idcs[idcs < 0] = 0
            idcs[idcs > cvec.size - 2] = cvec.size - 2
            index_vecs.append(idcs)

            norm_distances.append((xi - cvec[idcs]) /
                                  (cvec[idcs + 1] - cvec[idcs]))

        return index_vecs, norm_distances

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluation method, needs to be overridden."""
        raise NotImplementedError


class _NearestInterpolator(_Interpolator):

    """Nearest neighbor interpolator.

    The code is adapted from SciPy's `RegularGridInterpolator
    <http://docs.scipy.org/doc/scipy/reference/generated/\
scipy.interpolate.RegularGridInterpolator.html>`_ class.

    This implementation is faster than the more generic one in the
    `_PerAxisPointwiseInterpolator`. Compared to the original code,
    support of ``'left'`` and ``'right'`` variants are added.
    """

    def __init__(self, coord_vecs, values, input_type, variant):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`
            Coordinate vectors defining the interpolation grid
        values : array-like
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        variant : {'left', 'right'}
            Indicates which neighbor to prefer in the interpolation
        """
        super().__init__(coord_vecs, values, input_type)
        variant_ = str(variant).lower()
        if variant_ not in ('left', 'right'):
            raise ValueError("Variant '{}' not understood.".format(variant_))
        self.variant = variant_

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate nearest interpolation."""
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


def _compute_nearest_weights_edge(idcs, ndist, variant):
    """Helper for nearest interpolation mimicing the linear case."""
    # Get out-of-bounds indices from the norm_distances. Negative
    # means "too low", larger than or equal to 1 means "too high"
    lo = (ndist < 0)
    hi = (ndist > 1)

    # For "too low" nodes, the lower neighbor gets weight zero;
    # "too high" gets 1.
    if variant == 'left':
        w_lo = np.where(ndist <= 0.5, 1.0, 0.0)
    else:
        w_lo = np.where(ndist < 0.5, 1.0, 0.0)

    w_lo[lo] = 0
    w_lo[hi] = 1

    # For "too high" nodes, the upper neighbor gets weight zero;
    # "too low" gets 1.
    if variant == 'left':
        w_hi = np.where(ndist <= 0.5, 0.0, 1.0)
    else:
        w_hi = np.where(ndist < 0.5, 0.0, 1.0)

    w_hi[lo] = 1
    w_hi[hi] = 0

    # For upper/lower out-of-bounds nodes, we need to set the
    # lower/upper neighbors to the last/first grid point
    edge = [idcs, idcs + 1]
    edge[0][hi] = -1
    edge[1][lo] = 0

    return w_lo, w_hi, edge


def _compute_linear_weights_edge(idcs, ndist):
    """Helper for linear interpolation."""
    # Get out-of-bounds indices from the norm_distances. Negative
    # means "too low", larger than or equal to 1 means "too high"
    lo = np.where(ndist < 0)
    hi = np.where(ndist > 1)

    # For "too low" nodes, the lower neighbor gets weight zero;
    # "too high" gets 2 - yi (since yi >= 1)
    w_lo = (1 - ndist)
    w_lo[lo] = 0
    w_lo[hi] += 1

    # For "too high" nodes, the upper neighbor gets weight zero;
    # "too low" gets 1 + yi (since yi < 0)
    w_hi = np.copy(ndist)
    w_hi[lo] += 1
    w_hi[hi] = 0

    # For upper/lower out-of-bounds nodes, we need to set the
    # lower/upper neighbors to the last/first grid point
    edge = [idcs, idcs + 1]
    edge[0][hi] = -1
    edge[1][lo] = 0

    return w_lo, w_hi, edge


def _create_weight_edge_lists(indices, norm_distances, schemes, variants):
    # Precalculate indices and weights (per axis)
    low_weights = []
    high_weights = []
    edge_indices = []
    for i, (idcs, yi, scm, var) in enumerate(
            zip(indices, norm_distances, schemes, variants)):
        if scm == 'nearest':
            w_lo, w_hi, edge = _compute_nearest_weights_edge(
                idcs, yi, var)
        elif scm == 'linear':
            w_lo, w_hi, edge = _compute_linear_weights_edge(
                idcs, yi)
        else:
            raise ValueError("scheme '{}' at index {} not supported."
                             "".format(scm, i))

        low_weights.append(w_lo)
        high_weights.append(w_hi)
        edge_indices.append(edge)

    return low_weights, high_weights, edge_indices


class _PerAxisInterpolator(_Interpolator):

    """Interpolator where the scheme is set per axis.

    This allows to use e.g. nearest neighbor interpolation in the
    first dimension and linear in dimensions 2 and 3.
    """

    def __init__(self, coord_vecs, values, input_type, schemes, nn_variants):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`
            Coordinate vectors defining the interpolation grid
        values : array-like
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        schemes : sequence of `str`
            Indicates which interpolation scheme to use for which axis
        nn_variants : sequence of `str`
            Which variant ('left' or 'right') to use in nearest neighbor
            interpolation for which axis.
            This option has no effect for schemes other than nearest
            neighbor.
        """
        super().__init__(coord_vecs, values, input_type)
        self.schemes = schemes
        self.nn_variants = nn_variants

    def _evaluate(self, indices, norm_distances, out=None):
        """Evaluate linear interpolation.

        Modified for in-place evaluation and treatment of out-of-bounds
        points by implicitly assuming 0 at the next node."""
        # slice for broadcasting over trailing dimensions in self.values
        vslice = (slice(None),) + (None,) * (self.values.ndim - len(indices))

        out_shape = out_shape_from_meshgrid(norm_distances)
        out_dtype = norm_distances[0].dtype

        if out is None:
            out = np.zeros(out_shape, dtype=out_dtype)
        else:
            out[:] = 0.0

        # Weights and indices (per axis)
        low_weights, high_weights, edge_indices = _create_weight_edge_lists(
            indices, norm_distances, self.schemes, self.nn_variants)

        # Iterate over all possible combinations of [i, i+1] for each
        # axis, resulting in a loop of length 2**ndim
        for lo_hi, edge in zip(product(*([['l', 'h']] * len(indices))),
                               product(*edge_indices)):
            weight = 1.0
            for lh, w_lo, w_hi in zip(lo_hi, low_weights, high_weights):

                # We don't multiply in place to exploit the cheap operations
                # in the beginning: sizes grow gradually as following:
                # (n, 1, 1, ...) -> (n, m, 1, ...) -> ...
                # Hence, it is faster to build up the weight array instead
                # of doing full-size operations from the beginning.
                if lh == 'l':
                    weight = weight * w_lo
                else:
                    weight = weight * w_hi
            out += np.asarray(self.values[edge]) * weight[vslice]
        return np.array(out, copy=False, ndmin=1)


class _LinearInterpolator(_PerAxisInterpolator):

    """Linear (i.e. bi-/tri-/multi-linear) interpolator.

    Convenience class.
    """

    def __init__(self, coord_vecs, values, input_type):
        """Initialize a new instance.

        coord_vecs : sequence of `numpy.ndarray`
            Coordinate vectors defining the interpolation grid
        values : array-like
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        super().__init__(coord_vecs, values, input_type,
                         schemes=['linear'] * len(coord_vecs),
                         nn_variants=[None] * len(coord_vecs))


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
