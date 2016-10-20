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

from itertools import product
import numpy as np

from odl.operator import Operator
from odl.discr.partition import RectPartition
from odl.space.base_ntuples import NtuplesBase, FnBase
from odl.space import FunctionSet, FunctionSpace
from odl.util.vectorization import (
    is_valid_input_meshgrid, out_shape_from_array, out_shape_from_meshgrid)


__all__ = ('FunctionSetMapping',
           'PointCollocation', 'NearestInterpolation', 'LinearInterpolation',
           'PerAxisInterpolation')

_SUPPORTED_INTERP_SCHEMES = ['nearest', 'linear']


class FunctionSetMapping(Operator):

    """Abstract base class for function set discretization mappings."""

    def __init__(self, map_type, fset, partition, dspace, linear=False,
                 **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        map_type : {'sampling', 'interpolation'}
            The type of operator
        fset : `FunctionSet`
            The non-discretized (abstract) set of functions to be
            discretized
        partition : `RectPartition`
            Partition of (a subset of) ``fset.domain`` based on a
            `TensorGrid`
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points.
        linear : bool
            Create a linear operator if ``True``, otherwise a non-linear
            operator.
        order : {'C', 'F'}, optional
            Ordering of the axes in the data storage. 'C' means the
            first axis varies slowest, the last axis fastest;
            vice versa for 'F'.
            Default: 'C'
        """
        map_type_ = str(map_type).lower()
        if map_type_ not in ('sampling', 'interpolation'):
            raise ValueError('`map_type` {} not understood'
                             ''.format(map_type))
        if not isinstance(fset, FunctionSet):
            raise TypeError('`fset` {!r} is not a `FunctionSet` '
                            'instance'.format(fset))

        if not isinstance(partition, RectPartition):
            raise TypeError('`partition` {!r} is not a `RectPartition` '
                            'instance'.format(partition))
        if not isinstance(dspace, NtuplesBase):
            raise TypeError('`dspace` {!r} is not an `NtuplesBase` instance'
                            ''.format(dspace))

        if not fset.domain.contains_set(partition):
            raise ValueError('{} not contained in the domain {} '
                             'of the function set {}'
                             ''.format(partition, fset.domain, fset))

        if dspace.size != partition.size:
            raise ValueError('size {} of the data space {} not equal '
                             'to the size {} of the partition'
                             ''.format(dspace.size, dspace, partition.size))

        domain = fset if map_type_ == 'sampling' else dspace
        range = dspace if map_type_ == 'sampling' else fset
        Operator.__init__(self, domain, range, linear=linear)
        self.__partition = partition

        if self.is_linear:
            if not isinstance(fset, FunctionSpace):
                raise TypeError('`fset` {!r} is not a `FunctionSpace` '
                                'instance'.format(fset))
            if not isinstance(dspace, FnBase):
                raise TypeError('`dspace` {!r} is not an `FnBase` instance'
                                ''.format(dspace))
            if fset.field != dspace.field:
                raise ValueError('`field` {} of the function space and `field`'
                                 ' {} of the data space are not equal'
                                 ''.format(fset.field, dspace.field))

        order = str(kwargs.pop('order', 'C'))
        if str(order).upper() not in ('C', 'F'):
            raise ValueError('`order` {!r} not recognized'.format(order))
        else:
            self.__order = str(order).upper()

    def __eq__(self, other):
        """Return ``self == other``."""
        if self is other:
            return True
        else:
            return (isinstance(other, type(self)) and
                    isinstance(self, type(other)) and
                    self.domain == other.domain and
                    self.range == other.range and
                    self.partition == other.partition and
                    self.order == other.order)

    @property
    def partition(self):
        """Underlying domain partition."""
        return self.__partition

    @property
    def grid(self):
        """Sampling grid."""
        return self.partition.grid

    @property
    def order(self):
        """Axis ordering in the data storage."""
        return self.__order


class PointCollocation(FunctionSetMapping):

    """Function evaluation at grid points.

    This operator evaluates a given function in a set of points. These
    points are given as the sampling grid of a partition of the
    function domain. The result of this evaluation is an array of
    function values at these points.

    If, for example, a function is defined on the interval [0, 1],
    and a partition divides the interval into ``N`` subintervals,
    the resulting array will have length ``N``. The sampling points
    are defined by the partition, usually they are the midpoints
    of the subintervals.

    In higher dimensions, the same principle is applied, with the
    only difference being the additional information about the ordering
    of the axes in the flat storage array (C- vs. Fortran ordering).

    This operator is the default 'sampling' used by all core
    discretization classes.
    """

    def __init__(self, ip_fset, partition, dspace, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fset : `FunctionSet`
            The non-discretized (abstract) set of functions to be
            discretized. The function domain must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``ip_fset.domain`` based on a
            `TensorGrid`
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points.
        order : {'C', 'F'}, optional
            Ordering of the axes in the data storage. 'C' means the
            first axis varies slowest, the last axis fastest;
            vice versa for 'F'.
            Default: 'C'

        Examples
        --------
        Define a set of functions from the rectangle [1, 3] x [2, 5]
        to the real numbers:

        >>> rect = odl.IntervalProd([1, 3], [2, 5])
        >>> funcset = odl.FunctionSpace(rect)

        Partition the rectangle by a tensor grid:

        >>> rect = odl.IntervalProd([1, 3], [2, 5])
        >>> grid = odl.TensorGrid([1, 2], [3, 4, 5])
        >>> partition = odl.RectPartition(rect, grid)
        >>> rn = odl.rn(grid.size)

        Finally create the operator and test it on a function:

        >>> coll_op = PointCollocation(funcset, partition, rn)
        ...
        ... # Properly vectorized function
        >>> func_elem = funcset.element(lambda x: x[0] - x[1])
        >>> coll_op(func_elem)
        rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])
        >>> coll_op(lambda x: x[0] - x[1])  # Works directly
        rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])
        >>> out = odl.rn(6).element()
        >>> coll_op(func_elem, out=out)  # In-place
        rn(6).element([-2.0, -3.0, -4.0, -1.0, -2.0, -3.0])

        Fortran ordering:

        >>> coll_op = PointCollocation(funcset, partition, rn, order='F')
        >>> coll_op(func_elem)
        rn(6).element([-2.0, -1.0, -3.0, -2.0, -4.0, -3.0])
        """
        linear = isinstance(ip_fset, FunctionSpace)
        FunctionSetMapping.__init__(self, 'sampling', ip_fset, partition,
                                    dspace, linear, **kwargs)

    def _call(self, func, out=None, **kwargs):
        """Evaluate ``func`` at the grid of this operator.

        Parameters
        ----------
        func : `FunctionSetElement`
            The function to be evaluated
        out : `NtuplesBaseVector`, optional
            Array to which the values are written. Its shape must be
            ``(N,)``, where N is the total number of grid points. The
            data type must be the same as in the ``dspace`` of this
            mapping.
        kwargs :
            Additional keyword arguments, optional

        Returns
        -------
        out : `NtuplesBaseVector`, optional
            The function values at the grid points. If ``out`` was
            provided, the returned object is a reference to it.

        Notes
        -----
        This operator expects its input functions to be written in
        a vectorization-conforming manner to ensure fast evaluation.
        See the `vectorization guide
        <https://odlgroup.github.io/odl/guide/in_depth/\
vectorization_guide.html>`_ for a detailed introduction.

        See Also
        --------
        odl.discr.grid.TensorGrid.meshgrid
        numpy.meshgrid
        """
        mesh = self.grid.meshgrid
        if out is None:
            out = func(mesh, **kwargs).ravel(order=self.order)
        else:
            out[:] = np.ravel(
                func(mesh, out=out.asarray().reshape(self.grid.shape,
                                                     order=self.order),
                     **kwargs),
                order=self.order)

        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n  {!r},\n  {!r},\n  {!r}'.format(
            self.domain, self.grid, self.range)
        if self.order != 'C':
            inner_str += ",\n  order='{}'".format(self.order)
        return '{}({})'.format(self.__class__.__name__, inner_str)


class NearestInterpolation(FunctionSetMapping):

    """Nearest neighbor interpolation as an `Operator`.

    Given points ``x1 < x2 < ... < xN``, and values ``f1, ..., fN``,
    nearest neighbor interpolation at ``x`` is defined by::

        I(x) = fj  with j such that |x - xj| is minimal.

    The ambiguity at the midpoints is resolved by preferring one of the
    neighbors. For higher dimensions, this rule is applied per
    component.

    The nearest neighbor interpolation operator is defined as the
    mapping from the values ``f1, ..., fN`` to the function ``I(x)``
    (as a whole).

    In higher dimensions, this principle is applied per axis, the
    only difference being the additional information about the ordering
    of the axes in the flat storage array (C- vs. Fortran ordering).
    """

    def __init__(self, fset, partition, dspace, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fset : `FunctionSet`
            The undiscretized (abstract) set of functions to be
            discretized. The function domain must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``ip_fset.domain`` based on a
            spatial grid
        dspace : `NtuplesBase`
            Data space providing containers for the values of a
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points.

        Other Parameters
        ----------------
        variant : {'left', 'right'}, optional
            Behavior variant at midpoint between neighbors

            'left' : favor left neighbor (default)

            'right' : favor right neighbor
        order : {'C', 'F'}, optional
            Ordering of the axes in the data storage. 'C' means the
            first axis varies slowest, the last axis fastest;
            vice versa for 'F'.
            Default: 'C'

        Examples
        --------
        We test nearest neighbor interpolation with a non-scalar
        data type in 2d:

        >>> rect = odl.IntervalProd([0, 0], [1, 1])
        >>> strings = odl.Strings(1)  # 1-char strings
        >>> space = odl.FunctionSet(rect, strings)

        Partitioning the domain uniformly with no nodes on the boundary
        (will shift the grid points):

        >>> part = odl.uniform_partition_fromintv(rect, [4, 2],
        ...                                       nodes_on_bdry=False)
        >>> part.grid.coord_vectors
        (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))

        >>> dspace = odl.ntuples(part.size, dtype='U1')

        Now we initialize the operator and test it with some points:

        >>> interp_op = NearestInterpolation(space, part, dspace)
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

        Notes
        -----
        The distinction between 'left' and 'right' variants is currently
        made by changing ``<=`` to ``<`` at one place. This difference
        may not be noticable in some situations due to rounding errors.
        """
        linear = isinstance(fset, FunctionSpace)
        FunctionSetMapping.__init__(self, 'interpolation', fset, partition,
                                    dspace, linear, **kwargs)

        variant = kwargs.pop('variant', 'left')
        self.__variant = str(variant).lower()
        if self.variant not in ('left', 'right'):
            raise ValueError("`variant` '{}' not understood".format(variant))

    @property
    def variant(self):
        """The variant (left / right) of interpolation."""
        return self.__variant

    def _call(self, x, out=None):
        """Create an interpolator from grid values ``x``.

        Parameters
        ----------
        x : `NtuplesBaseVector`
            The array of values to be interpolated
        out : `FunctionSetElement`, optional
            Element in which to store the interpolator

        Returns
        -------
        out : `FunctionSetElement`
            Nearest-neighbor interpolator for the grid of this
            operator. If ``out`` was provided, the returned object
            is a reference to it.

        See Also
        --------
        LinearInterpolation : (bi-/tri-/...)linear interpolation

        Notes
        -----
        **Important:** if called on a point array, the points are
        assumed to be sorted in ascending order in each dimension
        for efficiency reasons.

        Nearest neighbor interpolation is the only scheme which works
        with data of non-scalar type since it does not involve any
        arithmetic operations on the values.
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
                x.asarray().reshape(self.grid.shape, order=self.order),
                variant=self.variant,
                input_type=input_type)

            return interpolator(arg, out=out)

        return self.range.element(nearest, vectorized=True)

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n  {!r},\n  {!r},\n  {!r}'.format(
            self.range, self.grid, self.domain)
        sep = ',\n '
        if self.order != 'C':
            inner_str += sep + "order='{}'".format(self.order)
            sep = ', '
        if self.variant != 'left':
            inner_str += sep + "variant='{}'".format(self.variant)

        return '{}({})'.format(self.__class__.__name__, inner_str)


class LinearInterpolation(FunctionSetMapping):

    """Linear interpolation interpolation as an `Operator`."""

    def __init__(self, fspace, partition, dspace, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The undiscretized (abstract) space of functions to be
            discretized. Its field must be the same as that of data
            space. The function domain must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `TensorGrid`
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points, and its `FnBase.field`
            must be the same as that of the function space.
        order : {'C', 'F'}, optional
            Ordering of the axes in the data storage. 'C' means the
            first axis varies slowest, the last axis fastest;
            vice versa for 'F'.
            Default: 'C'
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('`fspace` {!r} is not a `FunctionSpace` '
                            'instance'.format(fspace))

        FunctionSetMapping.__init__(self, 'interpolation', fspace, partition,
                                    dspace, linear=True, **kwargs)

    def _call(self, x, out=None):
        """Create an interpolator from grid values ``x``.

        Parameters
        ----------
        x : `FnBaseVector`
            The array of values to be interpolated
        out : `FunctionSpaceElement`, optional
            Element in which to store the interpolator

        Returns
        -------
        out : `FunctionSpaceElement`
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
                x.asarray().reshape(self.grid.shape, order=self.order),
                input_type=input_type)

            return interpolator(arg, out=out)

        return self.range.element(linear, vectorized=True)

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_str = '\n  {!r},\n  {!r},\n  {!r}'.format(self.range,
                                                        self.grid,
                                                        self.domain)
        if self.order != 'C':
            inner_str += ",\n  order='{}'".format(self.order)

        return '{}({})'.format(self.__class__.__name__, inner_str)


class PerAxisInterpolation(FunctionSetMapping):

    """Interpolation scheme set for each axis individually."""

    def __init__(self, fspace, partition, dspace, schemes, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The undiscretized (abstract) space of functions to be
            discretized. Its field must be the same as that of data
            space. The function domain must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `TensorGrid`
        dspace : `FnBase`
            Data space providing containers for the values of a
            discretized object. Its `NtuplesBase.size` must be equal
            to the total number of grid points, and its `FnBase.field`
            must be the same as that of the function space.
        schemes : string or sequence of strings
            Indicates which interpolation scheme to use for which axis.
            A single string is interpreted as a global scheme for all
            axes.
        nn_variants : string or sequence of strings, optional
            Which variant ('left' or 'right') to use in nearest neighbor
            interpolation for which axis. A single string is interpreted
            as a global variant for all axes.
            This option has no effect for schemes other than nearest
            neighbor.
            Default: 'left'
        order : {'C', 'F'}, optional
            Ordering of the axes in the data storage. 'C' means the
            first axis varies slowest, the last axis fastest;
            vice versa for 'F'.
            Default: 'C'
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('`fspace` {!r} is not a `FunctionSpace` '
                            'instance'.format(fspace))

        FunctionSetMapping.__init__(self, 'interpolation', fspace, partition,
                                    dspace, linear=True, **kwargs)

        try:
            schemes_ = str(schemes + '').lower()  # pythonic string check
            schemes_ = [schemes_] * self.grid.ndim
        except TypeError:
            schemes_ = [str(scm).lower() if scm is not None else None
                        for scm in schemes]

        nn_variants = kwargs.pop('nn_variants', None)
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
                raise ValueError("interpolation scheme '{}' at index {} not "
                                 "understood".format(scm, i))
            if scm == 'nearest' and var not in ('left', 'right'):
                raise ValueError("nearest neighbor variant '{}' at index {} "
                                 "not understood".format(var, i))
            elif scm != 'nearest' and var is not None:
                raise ValueError('option nn_variants used in axis {} with '
                                 'scheme {!r}'.format(i, scm))

        self.__schemes = schemes_
        self.__nn_variants = variants_

    @property
    def schemes(self):
        """List of interpolation schemes, one for each axis."""
        return self.__schemes

    @property
    def nn_variants(self):
        """List of nearest neighbor variants, one for each axis."""
        return self.__nn_variants

    def _call(self, x, out=None):
        """Create an interpolator from grid values ``x``.

        Parameters
        ----------
        x : `FnBaseVector`
            The array of values to be interpolated
        out : `FunctionSpaceElement`, optional
            Element in which to store the interpolator

        Returns
        -------
        out : `FunctionSpaceElement`
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
                x.asarray().reshape(self.grid.shape, order=self.order),
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

        inner_str = '\n {!r},\n {!r},\n {!r},\n {!r}'.format(
            self.range, self.grid, self.domain, schemes)
        sep = '\n, '
        if self.order != 'C':
            inner_str += sep + "order='{}'".format(self.order)
            sep = ', '

        if all(var == self.nn_variants[0] for var in self.nn_variants):
            variants = self.nn_variants[0]
        else:
            variants = self.nn_variants
            sep = ',\n '

        if variants is not None:
            inner_str += sep + 'nn_variants={}'.format(variants)

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

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        values = np.asarray(values)
        typ_ = str(input_type).lower()
        if typ_ not in ('array', 'meshgrid'):
            raise ValueError('`input_type` ({}) not understood'
                             ''.format(input_type))

        if len(coord_vecs) > values.ndim:
            raise ValueError('there are {} point arrays, but `values` has {} '
                             'dimensions'.format(len(coord_vecs),
                                                 values.ndim))
        for i, p in enumerate(coord_vecs):
            if not np.asarray(p).ndim == 1:
                raise ValueError('the points in dimension {} must be '
                                 '1-dimensional'.format(i))
            if values.shape[i] != len(p):
                raise ValueError('there are {} points and {} values in '
                                 'dimension {}'.format(len(p),
                                                       values.shape[i], i))

        self.coord_vecs = tuple(np.asarray(p) for p in coord_vecs)
        self.values = values
        self.input_type = input_type

    def __call__(self, x, out=None):
        """Do the interpolation.

        Parameters
        ----------
        x : `meshgrid` or `numpy.ndarray`
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
                                 'the grid dimension {}'
                                 ''.format(len(x), ndim))
            out_shape = out_shape_from_meshgrid(x)

        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError('`out` {!r} not a `numpy.ndarray` '
                                'instance'.format(out))
            if out.shape != out_shape:
                raise ValueError('output shape {} not equal to expected '
                                 'shape {}'.format(out.shape, out_shape))
            if out.dtype != self.values.dtype:
                raise ValueError('output dtype {} not equal to expected '
                                 'dtype {}'
                                 ''.format(out.dtype, self.values.dtype))

        indices, norm_distances = self._find_indices(x)
        return self._evaluate(indices, norm_distances, out)

    def _find_indices(self, x):
        """Find indices and distances of the given nodes.

        Can be overridden by subclasses to improve efficiency.
        """
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

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        variant : {'left', 'right'}
            Indicates which neighbor to prefer in the interpolation
        """
        super().__init__(coord_vecs, values, input_type)
        variant_ = str(variant).lower()
        if variant_ not in ('left', 'right'):
            raise ValueError("variant '{}' not understood".format(variant_))
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
            raise ValueError("scheme '{}' at index {} not supported"
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

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        schemes : sequence of strings
            Indicates which interpolation scheme to use for which axis
        nn_variants : sequence of strings
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

        if out is None:
            out_shape = out_shape_from_meshgrid(norm_distances)
            out_dtype = self.values.dtype
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
            # TODO: determine best summation order from array strides
            for lh, w_lo, w_hi in zip(lo_hi, low_weights, high_weights):

                # We don't multiply in-place to exploit the cheap operations
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

        coord_vecs : sequence of `numpy.ndarray`'s
            Coordinate vectors defining the interpolation grid
        values : `array-like`
            Grid values to use for interpolation
        input_type : {'array', 'meshgrid'}
            Type of expected input values in ``__call__``
        """
        super().__init__(coord_vecs, values, input_type,
                         schemes=['linear'] * len(coord_vecs),
                         nn_variants=[None] * len(coord_vecs))


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
