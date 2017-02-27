# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
from odl.space.base_tensors import TensorSpace
from odl.space import FunctionSpace
from odl.util import (
    is_valid_input_meshgrid, out_shape_from_array, out_shape_from_meshgrid,
    writable_array, signature_string, indent_rows, dtype_repr,
    is_numeric_dtype)


__all__ = ('FunctionSpaceMapping',
           'PointCollocation', 'NearestInterpolation', 'LinearInterpolation',
           'PerAxisInterpolation')

_SUPPORTED_INTERP_SCHEMES = ['nearest', 'linear']


class FunctionSpaceMapping(Operator):

    """Abstract base class for function set discretization mappings."""

    def __init__(self, map_type, fspace, partition, dspace, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        map_type : {'sampling', 'interpolation'}
            The type of operator
        fspace : `FunctionSpace`
            The non-discretized (abstract) set of functions to be
            discretized
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `RectGrid`.
        dspace : `TensorSpace`
            Data space providing containers for the values of a
        linear : bool, optional
            discretized object. Its `TensorSpace.shape` must be equal
            to ``partition.shape``.
        linear : bool
            Create a linear operator if ``True``, otherwise a non-linear
            operator.
        """
        map_type, map_type_in = str(map_type).lower(), map_type
        if map_type not in ('sampling', 'interpolation'):
            raise ValueError('`map_type` {!r} not understood'
                             ''.format(map_type_in))
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('`fspace` {!r} is not a `FunctionSpace` '
                            'instance'.format(fspace))

        if not isinstance(partition, RectPartition):
            raise TypeError('`partition` {!r} is not a `RectPartition` '
                            'instance'.format(partition))
        if not isinstance(dspace, TensorSpace):
            raise TypeError('`dspace` {!r} is not a `TensorSpace` instance'
                            ''.format(dspace))

        if not fspace.domain.contains_set(partition):
            raise ValueError('{} not contained in the domain {} '
                             'of the function set {}'
                             ''.format(partition, fspace.domain, fspace))

        if dspace.shape != partition.shape:
            raise ValueError('`dspace.shape` not equal to `partition.shape`: '
                             '{} != {}'
                             ''.format(dspace.shape, partition.shape))

        domain = fspace if map_type == 'sampling' else dspace
        range = dspace if map_type == 'sampling' else fspace
        Operator.__init__(self, domain, range, linear=linear)
        self.__partition = partition

        if self.is_linear:
            if self.domain.field is None:
                raise TypeError('`fspace.field` cannot be `None` for '
                                '`linear=True`')
            if not is_numerc_dtype(dspace.dtype):
                raise TypeError('`dspace.dtype` must be a numeric data type '
                                'for `linear=True`, got {}'
                                ''.format(dtype_repr(dspace)))
            if fspace.field != dspace.field:
                raise ValueError('`fspace.field` not equal to `dspace.field`: '
                                 '{} != {}'
                                 ''.format(fspace.field, dspace.field))

    def __eq__(self, other):
        """Return ``self == other``."""
        if self is other:
            return True
        else:
            return (isinstance(other, type(self)) and
                    isinstance(self, type(other)) and
                    self.domain == other.domain and
                    self.range == other.range and
                    self.partition == other.partition)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.domain, self.range, self.partition))

    @property
    def partition(self):
        """Underlying domain partition."""
        return self.__partition

    @property
    def grid(self):
        """Sampling grid."""
        return self.partition.grid


class PointCollocation(FunctionSpaceMapping):

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

    This operator is the default `DiscretizedSet.sampling` used by all
    core discretization classes.
    """

    def __init__(self, fspace, partition, dspace):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            Non-discretized (abstract) set of functions to be
            discretized. ``fspace.domain`` must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `RectGrid`.
        dspace : `TensorSpace`
            Data space providing containers for the values of a
            discretized object. Its `TensorSpace.shape` must be equal
            to ``partition.shape``.

        Examples
        --------
        Define a set of functions from the rectangle [1, 3] x [2, 5]
        to the real numbers:

        >>> rect = odl.IntervalProd([1, 3], [2, 5])
        >>> funcset = odl.FunctionSpace(rect)

        Partition the rectangle by a rectilinear grid:

        >>> rect = odl.IntervalProd([1, 3], [2, 5])
        >>> grid = odl.RectGrid([1, 2], [3, 4, 5])
        >>> partition = odl.RectPartition(rect, grid)
        >>> dspace = odl.rn(grid.shape)

        Finally create the operator and test it on a function:

        >>> coll_op = PointCollocation(funcset, partition, dspace)
        >>>
        >>> # Properly vectorized function
        >>> func_elem = funcset.element(lambda x: x[0] - x[1])
        >>> coll_op(func_elem)
        rn((2, 3)).element(
        [[-2.0, -3.0, -4.0],
         [-1.0, -2.0, -3.0]]
        )

        We can use a Python function directly without creating a
        function space element:

        >>> coll_op(lambda x: x[0] - x[1])
        rn((2, 3)).element(
        [[-2.0, -3.0, -4.0],
         [-1.0, -2.0, -3.0]]
        )

        Broadcasting and ``out`` parameters are supported:

        >>> out = dspace.element()
        >>> result = coll_op(func_elem, out=out)
        >>> result is out
        True
        >>> out
        rn((2, 3)).element(
        [[-2.0, -3.0, -4.0],
         [-1.0, -2.0, -3.0]]
        )

        It is possible to use parametric functions and pass the parameters
        during operator call. Currently, this must *always* happen through
        ``kwargs``:

        >>> def plus_c(x, **kwargs):
        ...     c = kwargs.pop('c', 0)
        ...     return x[0] - x[1] + c
        >>> coll_op(plus_c)  # uses default c = 0
        rn((2, 3)).element(
        [[-2.0, -3.0, -4.0],
         [-1.0, -2.0, -3.0]]
        )
        >>> coll_op(plus_c, c=2)
        rn((2, 3)).element(
        [[0.0, -1.0, -2.0],
         [1.0, 0.0, -1.0]]
        )

        Notes
        -----
        This operator expects its input functions to be written in
        a vectorization-conforming manner to ensure fast evaluation.
        See the `ODL vectorization guide`_ for a detailed introduction.

        See Also
        --------
        odl.discr.grid.RectGrid.meshgrid
        numpy.meshgrid

        References
        ----------
        .. _ODL vectorization guide:
           https://odlgroup.github.io/odl/guide/in_depth/\
vectorization_guide.html
        """
        linear = getattr(fspace, 'field', None) is not None
        FunctionSpaceMapping.__init__(self, 'sampling', fspace, partition,
                                      dspace, linear)

    def _call(self, func, out=None, **kwargs):
        """Return ``self(func[, out, **kwargs])``."""
        mesh = self.grid.meshgrid
        if out is None:
            out = func(mesh, **kwargs)
        else:
            with writable_array(out) as out_arr:
                func(mesh, out=out_arr, **kwargs)
        return out

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.range, self.grid, self.domain]
        inner_str = signature_string(posargs, [],
                                     sep=[',\n', ', ', ',\n'],
                                     mod=['!r', ''])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(inner_str))


class NearestInterpolation(FunctionSpaceMapping):

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

    def __init__(self, fspace, partition, dspace, variant='left'):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            Non-discretized (abstract) set of functions to be
            discretized. ``fspace.domain`` must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            spatial grid.
        dspace : `TensorSpace`
            Data space providing containers for the values of a
            discretized object. Its `TensorSpace.shape` must be equal
            to ``partition.shape``.
        variant : {'left', 'right'}, optional
            Behavior variant at the midpoint between neighbors.

                ``'left'``: favor left neighbor (default)

                ``'right'``: favor right neighbor

        Examples
        --------
        We test nearest neighbor interpolation with a non-scalar
        data type in 2d:

        >>> rect = odl.IntervalProd([0, 0], [1, 1])
        >>> strings = odl.Strings(1)  # 1-char strings
        >>> space = odl.FunctionSpace(rect, strings)

        Partitioning the domain uniformly with no nodes on the boundary
        (will shift the grid points):

        >>> part = odl.uniform_partition_fromintv(rect, [4, 2],
        ...                                       nodes_on_bdry=False)
        >>> part.grid.coord_vectors
        (array([ 0.125,  0.375,  0.625,  0.875]), array([ 0.25,  0.75]))

        >>> dspace = odl.tensor_space(part.shape, dtype='U1')

        Now we initialize the operator and test it with some points:

        >>> interp_op = NearestInterpolation(space, part, dspace)
        >>> values = np.array([c for c in 'mystring']).reshape(dspace.shape)
        >>> function = interp_op(values)
        >>> print(function([0.3, 0.6]))  # closest to index (1, 1) -> 3
        t
        >>> out = np.empty(2, dtype='U1')
        >>> pts = np.array([[0.3, 0.6],
        ...                 [1.0, 1.0]])
        >>> out = function(pts.T, out=out)  # returns original out
        >>> all(out == ['t', 'g'])
        True

        See Also
        --------
        LinearInterpolation : (bi-/tri-/...)linear interpolation

        Notes
        -----
        - **Important:** if called on a point array, the points are
          assumed to be sorted in ascending order in each dimension
          for efficiency reasons.
        - Nearest neighbor interpolation is the only scheme which works
          with data of non-scalar type since it does not involve any
          arithmetic operations on the values.
        - The distinction between 'left' and 'right' variants is currently
          made by changing ``<=`` to ``<`` at one place. This difference
          may not be noticable in some situations due to rounding errors.
        """
        linear = getattr(fspace, 'field', None) is not None
        FunctionSpaceMapping.__init__(self, 'interpolation', fspace, partition,
                                      dspace, linear)

        self.__variant = str(variant).lower()
        if self.variant not in ('left', 'right'):
            raise ValueError("`variant` {!r} not understood".format(variant))

    @property
    def variant(self):
        """The variant (left / right) of interpolation."""
        return self.__variant

    def _call(self, x, out=None):
        """Return ``self(x[, out])``."""
        # TODO: pass reasonable options on to the interpolator
        def nearest(arg, out=None):
            """Interpolating function with vectorization."""
            if is_valid_input_meshgrid(arg, self.grid.ndim):
                input_type = 'meshgrid'
            else:
                input_type = 'array'

            interpolator = _NearestInterpolator(
                self.grid.coord_vectors, x.asarray(), variant=self.variant,
                input_type=input_type)

            return interpolator(arg, out=out)

        return self.range.element(nearest, vectorized=True)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.range, self.grid, self.domain]
        optargs = [('variant', self.variant, 'left')]
        inner_str = signature_string(posargs, optargs,
                                     sep=[',\n', ', ', ',\n'],
                                     mod=['!r', ''])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(inner_str))


class LinearInterpolation(FunctionSpaceMapping):

    """Linear interpolation interpolation as an `Operator`."""

    def __init__(self, fspace, partition, dspace):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            Non-discretized (abstract) space of functions to be
            discretized. ``fspace.domain`` must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `RectGrid`
        dspace : `TensorSpace`
            Data space providing containers for the values of a
            discretized object. Its `TensorSpace.shape` must be equal
            to ``partition.shape``, and its `TensorSpace.field` must
            match ``fspace.field``.
        """
        if getattr(fspace, 'field', None) is None:
            raise TypeError('`fspace.field` cannot be `None`')
        FunctionSpaceMapping.__init__(self, 'interpolation', fspace, partition,
                                      dspace, linear=True)

    def _call(self, x, out=None):
        """Return ``self(x[, out])``."""
        # TODO: pass reasonable options on to the interpolator
        def linear(arg, out=None):
            """Interpolating function with vectorization."""
            if is_valid_input_meshgrid(arg, self.grid.ndim):
                input_type = 'meshgrid'
            else:
                input_type = 'array'

            interpolator = _LinearInterpolator(
                self.grid.coord_vectors, x.asarray(), input_type=input_type)

            return interpolator(arg, out=out)

        return self.range.element(linear, vectorized=True)

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.range, self.grid, self.domain]
        inner_str = signature_string(posargs, [],
                                     sep=[',\n', ', ', ',\n'],
                                     mod=['!r', ''])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(inner_str))


class PerAxisInterpolation(FunctionSpaceMapping):

    """Interpolation scheme set for each axis individually."""

    def __init__(self, fspace, partition, dspace, schemes, nn_variants='left'):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            Non-discretized (abstract) space of functions to be
            discretized. ``fspace.domain`` must provide a
            `Set.contains_set` method.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `RectGrid`
        dspace : `TensorSpace`
            Data space providing containers for the values of a
            discretized object. Its `TensorSpace.shape` must be equal
            to ``partition.shape``, and its `TensorSpace.field` must
            match ``fspace.field``.
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
        """
        if getattr(fspace, 'field', None) is None:
            raise TypeError('`fspace.field` cannot be `None`')
        FunctionSpaceMapping.__init__(self, 'interpolation', fspace, partition,
                                      dspace, linear=True)

        try:
            schemes_ = str(schemes + '').lower()  # pythonic string check
            schemes_ = [schemes_] * self.grid.ndim
        except TypeError:
            schemes_ = [str(scm).lower() if scm is not None else None
                        for scm in schemes]

        try:
            nn_variants + ''
        except TypeError:
            # No string
            variants_ = [str(var).lower() if var is not None else None
                         for var in nn_variants]
        else:
            # String
            variant = str(nn_variants).lower()
            variants_ = [variant if scm == 'nearest' else None
                         for scm in schemes_]

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
        x : `Tensor`
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
                x.asarray(),
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

        posargs = [self.range, self.grid, self.domain, schemes]

        nontriv_var = None
        for var in self.nn_variants:
            if var is not None:
                nontriv_var = var
                break

        if nontriv_var is None:
            variants = 'left'
        elif all(var == nontriv_var
                 for var in self.nn_variants
                 if var is not None):
            variants = nontriv_var
        else:
            variants = self.nn_variants

        optargs = [('nn_variants', variants, 'left')]

        inner_str = signature_string(posargs, optargs,
                                     sep=[',\n', ', ', ',\n'],
                                     mod=['!r', ''])
        return '{}(\n{}\n)'.format(self.__class__.__name__,
                                   indent_rows(inner_str))


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
        raise NotImplementedError('abstract method')


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
