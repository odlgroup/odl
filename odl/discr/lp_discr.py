# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Lebesgue L^p type discretizations of function spaces."""

from __future__ import print_function, division, absolute_import
from builtins import int
import numpy as np

from odl.discr.discretization import (
    DiscretizedSpace, DiscretizedSpaceElement, tspace_type)
from odl.discr.discr_mappings import (
    PointCollocation, NearestInterpolation, LinearInterpolation,
    PerAxisInterpolation)
from odl.discr.grid import uniform_grid
from odl.discr.partition import (
    RectPartition, uniform_partition_fromintv, uniform_partition_fromgrid,
    uniform_partition)
from odl.set import RealNumbers, ComplexNumbers, IntervalProd
from odl.space import FunctionSpace, ProductSpace
from odl.space.entry_points import tensor_space_impl
from odl.space.weighting import ConstWeighting, PerAxisWeighting
from odl.util import (
    apply_on_boundary, is_real_dtype, is_complex_floating_dtype, is_string,
    is_floating_dtype, is_numeric_dtype, is_int,
    dtype_str, array_str, signature_string, indent, npy_printoptions,
    normalized_scalar_param_list, safe_int_conv, normalized_nodes_on_bdry,
    normalized_index_expression, simulate_slicing)

__all__ = ('DiscreteLp', 'DiscreteLpElement',
           'uniform_discr_frompartition', 'uniform_discr_fromspace',
           'uniform_discr_fromintv', 'uniform_discr',
           'uniform_discr_fromdiscr')

_SUPPORTED_INTERP = ('nearest', 'linear')


class DiscreteLp(DiscretizedSpace):

    """Discretization of a Lebesgue :math:`L^p` space."""

    def __init__(self, fspace, partition, tspace, interp='nearest', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The continuous space to be discretized.
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain``.
        tspace : `TensorSpace`
            Space of elements used for data storage. It must have the
            same `TensorSpace.field` as ``fspace`` and the same
            `TensorSpace.shape` as ``partition``.
        interp : str or sequence of str, optional
            Interpolation type to be used for discretization.
            A sequence is interpreted as interpolation scheme per axis.
            Possible values:

            - ``'nearest'`` : use nearest-neighbor interpolation.
            - ``'linear'`` : use linear interpolation.

        axis_labels : sequence of str, optional
            Names of the axes to use for plotting etc.
            Default:

            - 1D: ``['$x$']``
            - 2D: ``['$x$', '$y$']``
            - 3D: ``['$x$', '$y$', '$z$']``
            - nD: ``['$x_1$', '$x_2$', ..., '$x_n$']``

            Note: The ``$`` signs ensure rendering as LaTeX.
        """
        if not isinstance(fspace, FunctionSpace):
            raise TypeError('{!r} is not a FunctionSpace instance'
                            ''.format(fspace))
        if not isinstance(fspace.domain, IntervalProd):
            raise TypeError('function space domain {!r} is not an '
                            'IntervalProd instance'.format(fspace.domain))
        if not isinstance(partition, RectPartition):
            raise TypeError('`partition` {!r} is not a RectPartition '
                            'instance'.format(partition))
        if not fspace.domain.contains_set(partition.set):
            raise ValueError('`partition` {} is not a subset of the function '
                             'domain {}'.format(partition, fspace.domain))
        if fspace.scalar_out_dtype != tspace.dtype:
            raise ValueError('`fspace.scalar_out_dtype` does not match '
                             '`tspace.dtype`: {} != {}'
                             ''.format(fspace.scalar_out_dtype, tspace.dtype))

        self.__partition = partition

        # Handle interp
        if is_string(interp):
            interp, interp_in = str(interp).lower(), interp
            if interp not in _SUPPORTED_INTERP:
                raise ValueError('`interp` {!r} not understood'
                                 ''.format(interp_in))
            # Ensure that there is 1 entry for ndim == 0
            self.__interp_byaxis = (interp,) * max(partition.ndim, 1)
        else:
            # Got sequence of strings
            if len(interp) != partition.ndim:
                raise ValueError('expected {} entries in `interp`, got {}'
                                 ''.format(partition.ndim, len(interp)))

            self.__interp_byaxis = tuple(str(s).lower() for s in interp)
            if any(s not in _SUPPORTED_INTERP for s in self.interp_byaxis):
                raise ValueError('`interp` sequence {} contains illegal '
                                 'values'.format(interp))

        # Assign sampling and interpolation operators
        sampling = PointCollocation(fspace, self.partition, tspace)
        if all(s == 'nearest' for s in self.interp_byaxis):
            interpol = NearestInterpolation(fspace, self.partition, tspace)
        elif all(s == 'linear' for s in self.interp_byaxis):
            interpol = LinearInterpolation(fspace, self.partition, tspace)
        else:
            interpol = PerAxisInterpolation(
                fspace, self.partition, tspace, self.interp_byaxis)

        super(DiscreteLp, self).__init__(fspace, tspace, sampling, interpol)

        # Set axis labels
        axis_labels = kwargs.pop('axis_labels', None)
        if axis_labels is None:
            axis_labels_out = tuple('$i_{}$'.format(ax)
                                    for ax in range(self.ndim_out))
            if self.ndim_in <= 3:
                axis_labels_in = ('$x$', '$y$', '$z$')[:self.ndim_in]
            else:
                axis_labels_in = tuple('$x_{}$'.format(ax)
                                       for ax in range(self.ndim_in))
            self.__axis_labels = axis_labels_out + axis_labels_in
        else:
            if len(axis_labels) != self.ndim:
                raise ValueError(
                    '`axis_labels` should be equal to the number of axes, '
                    'but {} != {}'.format(len(axis_labels), self.ndim))
            self.__axis_labels = tuple(str(label) for label in axis_labels)

        if kwargs:
            raise ValueError('got unexpected keyword arguments {}'
                             ''.format(kwargs))

    @property
    def interp(self):
        """Interpolation type of this discretization."""
        if self.ndim == 0:
            return 'nearest'
        elif all(interp == self.interp_byaxis[0]
                 for interp in self.interp_byaxis):
            return self.interp_byaxis[0]
        else:
            return self.interp_byaxis

    @property
    def interp_byaxis(self):
        """Interpolation by axis type of this discretization."""
        return self.__interp_byaxis

    @property
    def axis_labels(self):
        """Labels for axes when displaying space elements."""
        return self.__axis_labels

    @property
    def partition(self):
        """`RectPartition` of the function domain."""
        return self.__partition

    @property
    def exponent(self):
        """Exponent of this space, the ``p`` in ``L^p``."""
        return self.tspace.exponent

    @property
    def min_pt(self):
        """Vector of minimal coordinates of the function domain."""
        return self.partition.min_pt

    @property
    def max_pt(self):
        """Vector of maximal coordinates of the function domain."""
        return self.partition.max_pt

    @property
    def is_uniform_byaxis(self):
        """Boolean tuple showing uniformity of ``self.partition`` per axis."""
        return self.partition.is_uniform_byaxis

    @property
    def is_uniform(self):
        """``True`` if `partition` is uniform."""
        return self.partition.is_uniform

    @property
    def grid(self):
        """Sampling grid of the discretization mappings."""
        return self.partition.grid

    @property
    def shape(self):
        """Shape of the underlying tensor space.

        For spaces of discretized vector- or tensor-valued functions,
        this includes the output components as ::

            shape = fspace.out_shape + partition.shape
        """
        return self.tspace.shape

    @property
    def shape_in(self):
        """Shape of the "input" part, i.e. of the `partition`.

        For spaces of discretized scalar-valued functions, this is the
        same as `shape`. For vector- or tensor-valued functions, this
        is the shape of *one component*.
        """
        return self.partition.shape

    @property
    def ndim_in(self):
        """Number of "input" dimensions, equals ``len(self.shape_in)``."""
        return len(self.shape_in)

    @property
    def shape_out(self):
        """Shape of the "output" part, i.e. of the `fspace`.

        For spaces of discretized vector- or tensor-valued functions,
        this is the shape of a function value; for scalar-valued functions,
        it is an empty tuple.
        """
        return self.fspace.out_shape

    @property
    def ndim_out(self):
        """Number of "output" dimensions, equals ``len(self.shape_out)``."""
        return len(self.shape_out)

    @property
    def ndim(self):
        """Total number of dimensions, including input and output."""
        return self.ndim_in + self.ndim_out

    @property
    def dtype(self):
        """Data type of this space.

        For spaces of discretized vector- or tensor-valued functions,
        the data type has a shape. For the base data type without shape,
        use `scalar_dtype`.
        """
        # Construct this way since `fspace` may have dtype `None`
        return np.dtype((self.tspace.dtype, self.fspace.out_shape))

    @property
    def scalar_dtype(self):
        """Scalar data type of this space.

        For spaces of discretized vector- or tensor-valued functions,
        this is the variant of `dtype` without a shape.
        """
        return self.tspace.dtype

    @property
    def size(self):
        """Total number of entries in the underlying tensor space."""
        return self.tspace.size

    @property
    def cell_sides(self):
        """Side lengths of a cell in an underlying *uniform* partition."""
        return self.partition.cell_sides

    @property
    def cell_volume(self):
        """Cell volume of an underlying *uniform* partition."""
        return self.partition.cell_volume

    @property
    def meshgrid(self):
        """All sampling points in the partition as a sparse meshgrid."""
        return self.partition.meshgrid

    def points(self, order='C'):
        """All sampling points in the partition.

        Parameters
        ----------
        order : {'C', 'F'}
            Axis ordering in the resulting point array.

        Returns
        -------
        points : `numpy.ndarray`
            The shape of the array is ``size x ndim``, i.e. the points
            are stored as rows.
        """
        return self.partition.points(order)

    @property
    def default_order(self):
        """Default storage order for new elements in this space.

        This is equal to the default order of `tspace`.
        """
        return self.tspace.default_order

    @property
    def tangent_bundle(self):
        """The tangent bundle associated with `domain` using `partition`.

        The tangent bundle of a space ``X`` of functions ``R^d --> F`` can be
        interpreted as the space of vector-valued functions ``R^d --> F^d``.
        This space can be identified with the power space ``X^d`` as used
        in this implementation.
        """
        if self.ndim_in == 0:
            return ProductSpace(field=self.field)
        else:
            return ProductSpace(self, self.ndim_in)

    @property
    def is_uniformly_weighted(self):
        """``True`` if the weighting is the same for all space points."""
        try:
            is_uniformly_weighted = self.__is_uniformly_weighted
        except AttributeError:
            bdry_fracs = self.partition.boundary_cell_fractions
            is_uniformly_weighted = (
                np.allclose(bdry_fracs, 1.0) or
                self.exponent == float('inf') or
                not getattr(self.tspace, 'is_weighted', False))

            self.__is_uniformly_weighted = is_uniformly_weighted

        return is_uniformly_weighted

    def element(self, inp=None, order=None, **kwargs):
        """Create an element from ``inp`` or from scratch.

        Parameters
        ----------
        inp : optional
            Input used to initialize the new element. The following options
            are available:

            - ``None``: an empty element is created with no guarantee of
              its state (memory allocation only). The new element will
              use ``order`` as storage order if provided, otherwise
              `default_order`.

            - array-like: an element wrapping a `tensor` is created,
              where a copy is avoided whenever possible. This usually
              requires correct `shape`, `dtype` and `impl` if applicable,
              and if ``order`` is provided, also contiguousness in that
              ordering. See the ``element`` method of `tspace` for more
              information.

              If any of these conditions is not met, a copy is made.

            - callable: a new element is created by sampling the function
              using the `sampling` operator.

        order : {None, 'C', 'F'}, optional
            Storage order of the returned element. For ``'C'`` and ``'F'``,
            contiguous memory in the respective ordering is enforced.
            The default ``None`` enforces no contiguousness.
        vectorized : bool, optional
            If ``True``, assume that a provided callable ``inp`` supports
            vectorized evaluation. Otherwise, wrap it in a vectorizer.
            Default: ``True``.
        kwargs :
            Additional arguments passed on to `sampling` when called
            on ``inp``, in the form ``sampling(inp, **kwargs)``.
            This can be used e.g. for functions with parameters.

        Returns
        -------
        element : `DiscreteLpElement`
            The discretized element, calculated as ``sampling(inp)`` or
            ``tspace.element(inp)``, tried in this order.

        Examples
        --------
        Elements can be created from array-like objects that represent
        an already discretized function:

        >>> space = odl.uniform_discr(-1, 1, 4)
        >>> space.element([1, 2, 3, 4])
        uniform_discr(-1.0, 1.0, 4).element([ 1.,  2.,  3.,  4.])
        >>> vector = odl.rn(4).element([0, 1, 2, 3])
        >>> space.element(vector)
        uniform_discr(-1.0, 1.0, 4).element([ 0.,  1.,  2.,  3.])

        On the other hand, non-discretized objects like Python functions
        can be discretized "on the fly":

        >>> space.element(lambda x: x * 2)
        uniform_discr(-1.0, 1.0, 4).element([-1.5, -0.5, 0.5, 1.5])

        This works also with parameterized functions, however only
        through keyword arguments (not positional arguments with
        defaults):

        >>> def f(x, c=0.0):
        ...     return np.maximum(x, c)
        ...
        >>> space = odl.uniform_discr(-1, 1, 4)
        >>> space.element(f, c=0.5)
        uniform_discr(-1.0, 1.0, 4).element([ 0.5 ,  0.5 ,  0.5 ,  0.75])

        See Also
        --------
        sampling : create a discrete element from a non-discretized one
        """
        if inp is None:
            return self.element_type(self, self.tspace.element(order=order))
        elif inp in self and order is None:
            return inp
        elif inp in self.tspace and order is None:
            return self.element_type(self, inp)
        elif callable(inp):
            vectorized = kwargs.pop('vectorized', True)
            # fspace element -> discretize
            inp_elem = self.fspace.element(inp, vectorized=vectorized)
            sampled = self.sampling(inp_elem, **kwargs)
            return self.element_type(
                self, self.tspace.element(sampled, order=order))
        else:
            # Sequence-type input
            return self.element_type(
                self, self.tspace.element(inp, order=order))

    def _astype(self, dtype):
        """Internal helper for ``astype``."""
        dtype = np.dtype(dtype)
        fspace = self.fspace.astype(dtype)

        # If `dtype` has a shape, we remove the old output axes and insert
        # the new ones
        if dtype.shape == ():
            tspace = self.tspace.astype(dtype)
            axis_labels_new = self.axis_labels
        else:
            tspace = self.tspace.byaxis[self.ndim_out:].astype(dtype)
            axis_labels_extra = tuple('$i_{}$'.format(ax)
                                      for ax in range(len(dtype.shape)))
            axis_labels_new = (axis_labels_extra +
                               self.axis_labels[self.ndim_out:])

        return type(self)(
            fspace, self.partition, tspace, interp=self.interp,
            axis_labels=axis_labels_new)

    # Overrides for space functions depending on partition
    #
    # The inherited methods by default use a weighting by a constant
    # (the grid cell size). In dimensions where the partitioned set contains
    # only a fraction of the outermost cells (e.g. if the outermost grid
    # points lie at the boundary), the corresponding contribuitons to
    # discretized integrals need to be scaled by that fraction.
    def _inner(self, x, y):
        """Return ``self.inner(x, y)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            # TODO: implement without copying x
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=1.0)
            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return super(DiscreteLp, self)._inner(self.element(x_arr), y)
        else:
            return super(DiscreteLp, self)._inner(x, y)

    def _norm(self, x):
        """Return ``self.norm(x)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            # TODO: implement without copying x
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)
            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return super(DiscreteLp, self)._norm(self.element(x_arr))
        else:
            return super(DiscreteLp, self)._norm(x)

    def _dist(self, x, y):
        """Return ``self.dist(x, y)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)
            arrs = [apply_on_boundary(vec, func=func_list, only_once=False)
                    for vec in (x, y)]

            return super(DiscreteLp, self)._dist(
                self.element(arrs[0]), self.element(arrs[1]))
        else:
            return super(DiscreteLp, self)._dist(x, y)

    # TODO: add byaxis_out when discretized tensor-valued functions are
    # available

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        For most cases, indexing is implemented such that for an element
        ``x in space``, it holds ::

            x[indices] in space[indices]

        Space indexing does not work with index arrays or "fancy indexing".

        Notes
        -----
        New axes can be created, however not within the axes that
        correspond to the function domain, only as new output axes.
        In other words, a scalar function can be turned into a
        ``(1, ..., 1)`` tensor-valued function this way by adding new axes
        of size 1, or a vector-valued function into a tensor-valued one
        with sizes 1 in the new axes. This is useful for broadcasting,
        see Examples.

        Examples
        --------
        A single integer slices along the first axis (index does not
        matter as long as it lies within the bounds):

        >>> space = odl.uniform_discr([0, 0, 0, 0], [1, 2, 3, 4],
        ...                           shape=(2, 4, 6, 8))
        >>> space[0]
        uniform_discr([  0.,  0.,  0.], [  2.,  3.,  4.], (4, 6, 8))
        >>> space[1]
        uniform_discr([  0.,  0.,  0.], [  2.,  3.,  4.], (4, 6, 8))

        Multiple indices slice into corresponding axes from the left:

        >>> space[0, 0]
        uniform_discr([  0.,  0.], [  3.,  4.], (6, 8))
        >>> space[0, 1, 1]
        uniform_discr(0.0, 4.0, 8)

        Ellipsis (``...``) and ``slice(None)`` (``:``) can be used to keep
        one or several axes intact:

        >>> space[0, :, 1, :]
        uniform_discr([  0.,  0.], [  2.,  4.], (4, 8))
        >>> space[..., 0, 0]
        uniform_discr([ 0.,  0.], [ 1.,  2.], (2, 4))

        With slices, parts of axes can be selected:

        >>> space[0, :3, 1:4, :]
        uniform_discr([ 0. ,  0.5,  0. ], [ 1.5,  2. ,  4. ], (3, 3, 8))

        Indexing is also supported for vector- or tensor-valued function
        spaces. In that case, leftmost indices are associated with the
        output dimensions, rightmost indices with the input (domain)
        dimensions:

        >>> space = odl.uniform_discr(0, 1, 5, dtype=(float, (2, 3)))
        >>> space.shape
        (2, 3, 5)
        >>> space[0]  # (2, 3)-valued -> (3,)-valued (first row)
        uniform_discr(0.0, 1.0, 5, dtype=('float64', (3,)))
        >>> space[:, 0]  # (2, 3)-valued -> (2,)-valued (first column)
        uniform_discr(0.0, 1.0, 5, dtype=('float64', (2,)))
        >>> space[..., :2]  # last axis = domain
        uniform_discr(0.0, 0.4, 2, dtype=('float64', (2, 3)))
        """
        # Unwrap tensor for passing on to underlying indexing methods
        if isinstance(indices, self.element_type):
            indices = indices.tensor

        if getattr(indices, 'dtype', object) == bool:
            # Raising here as signal for `DiscreteLpElement.__getitem__`
            # This is an optimization that avoids counting the number of
            # `True` entries here *and* in the actual element indexing
            # operation. We do it lazily instead.
            raise TypeError('cannot index space with a boolean element or '
                            'array')

        # The overall logic of the slicing is as follows:
        # - indices are normalized to be a tuple of length `ndim`
        # - the `tspace` is indexed with the whole index tuple
        # - the `partition` is indexed with the "in" part of the tuple
        #   only, i.e., the last part excluding `n_out = len(shape_out)`
        #   axes
        # - the `out_shape` for the function space is determined using the
        #   first `n_out` index tuple entries
        # - the new function space is constructed from `new_partition.set`,
        #   `fspace.scalar_out_dtype` and the new `out_shape`

        # Normalize the index expression based on the full shape, and
        # split into "in" and "out" parts
        indices = normalized_index_expression(indices, self.shape)

        # New axes can be added to `indices_out`, so we can only use
        # `indices_in` as fixed in length
        indices_out = indices[:len(indices) - self.ndim_in]
        indices_in = indices[len(indices) - self.ndim_in:]
        # Now `indices_in` shouldn't contain any `None` since we don't
        # allow new axes there; we cannot use `indices.contains()` for
        # that check since it uses the `==` operator which will fail if
        # one of the indices is a Numpy array
        if any(idx is None for idx in indices_in):
            raise ValueError('new axes can be created only with respect to '
                             'the "output" dimensions, see the documentation')

        # Index tensor space
        res_tspace = self.tspace[indices]

        # Index partition, manually removing collapsed axes
        res_part = self.partition[indices_in]
        _, collapsed_axes_in, _, _ = simulate_slicing(
            self.shape_in, indices_in)
        remaining_axes_in = [i for i in range(len(self.shape_in))
                             if i not in collapsed_axes_in]
        res_part = res_part.byaxis[remaining_axes_in]

        # Determine new fspace
        sliced_shape_out, _, _, _ = simulate_slicing(
            self.shape_out, indices_out)
        res_fspace = FunctionSpace(
            res_part.set,
            out_dtype=(self.fspace.scalar_out_dtype, sliced_shape_out))

        # Further attributes for the new space
        res_interp = [self.interp_byaxis[i] for i in range(len(self.shape_in))
                      if i in remaining_axes_in]
        res_labels = [self.axis_labels[i] for i in range(len(self.shape_in))
                      if i in remaining_axes_in]

        # Create new space
        return DiscreteLp(res_fspace, res_part, res_tspace, interp=res_interp,
                          axis_labels=res_labels)

    @property
    def byaxis_in(self):
        """Object to index along input (domain) dimensions.

        Examples
        --------
        Indexing with integers or slices:

        >>> space = odl.uniform_discr([0, 0, 0], [1, 2, 3], (5, 10, 15))
        >>> space.byaxis_in[0]
        uniform_discr(0.0, 1.0, 5)
        >>> space.byaxis_in[1]
        uniform_discr(0.0, 2.0, 10)
        >>> space.byaxis_in[1:]
        uniform_discr([ 0.,  0.], [ 2.,  3.], (10, 15))

        Lists can be used to stack spaces arbitrarily:

        >>> space.byaxis_in[[2, 1, 2]]
        uniform_discr([ 0.,  0.,  0.], [ 3.,  2.,  3.], (15, 10, 15))
        """
        space = self

        class DiscreteLpByaxisIn(object):

            """Helper class for indexing by domain axes."""

            def __getitem__(self, indices):
                """Return ``self[indices]``.

                Parameters
                ----------
                indices : index expression
                    Object used to index the space domain.

                Returns
                -------
                space : `DiscreteLp`
                    The resulting space with indexed domain and otherwise
                    same properties (except possibly weighting).
                """
                fspace = space.fspace.byaxis_in[indices]
                part = space.partition.byaxis[indices]
                tspace = space.tspace.byaxis[indices]

                try:
                    iter(indices)
                except TypeError:
                    interp = space.interp_byaxis[indices]
                    labels = space.axis_labels[indices]
                else:
                    interp = tuple(space.interp_byaxis[int(i)]
                                   for i in indices)
                    labels = tuple(space.axis_labels[int(i)]
                                   for i in indices)

                return DiscreteLp(fspace, part, tspace, interp,
                                  axis_labels=labels)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis_in'

        return DiscreteLpByaxisIn()

    def __repr__(self):
        """Return ``repr(self)``."""
        # Clunky check if the factory repr can be used
        if (uniform_partition_fromintv(
                self.fspace.domain, self.shape_in,
                nodes_on_bdry=False) == self.partition):
            use_uniform = True
            nodes_on_bdry = False
        elif (uniform_partition_fromintv(
                self.fspace.domain, self.shape_in,
                nodes_on_bdry=True) == self.partition):
            use_uniform = True
            nodes_on_bdry = True
        else:
            use_uniform = False

        if use_uniform:
            ctor = 'uniform_discr'
            if self.ndim_in == 1:
                posargs = [self.min_pt[0], self.max_pt[0], self.shape_in[0]]
                posmod = [':.4', ':.4', '']
            else:
                posargs = [self.min_pt, self.max_pt, self.shape_in]
                posmod = [array_str, array_str, '']

            default_dtype_s = dtype_str(
                self.tspace.default_dtype(RealNumbers()))
            dtype_s = dtype_str(self.dtype)
            optargs = [('dtype', dtype_s, default_dtype_s),
                       ('interp', self.interp, 'nearest'),
                       ('impl', self.impl, 'numpy'),
                       ('nodes_on_bdry', nodes_on_bdry, False)]

            # Add weighting stuff if not equal to default
            default_facs_out = [1.0] * self.ndim_out
            if self.exponent != 2.0:
                weighting_part = self.weighting.repr_part

            elif (self.exponent == float('inf') or
                  self.ndim_in == 0 or
                  not is_floating_dtype(self.dtype)):
                # In these cases, weighting factors 1 is the default
                default_facs_in = [1.0] * self.ndim_in
                default_facs = default_facs_out + default_facs_in
                if (not isinstance(self.weighting, PerAxisWeighting) or
                        all(fac.ndim == 0 and np.isclose(fac, default_facs[i])
                            for i, fac in enumerate(self.weighting.factors))):
                    weighting_part = self.weighting.repr_part

            else:
                # Here, per-axis weighting with the cell sides as factors is
                # the default
                default_facs_in = list(self.cell_sides)
                default_facs = default_facs_out + default_facs_in

                if isinstance(self.weighting, ConstWeighting):
                    weighting_part = ('weighting={:.4}'
                                      ''.format(self.weighting.const))
                elif (isinstance(self.weighting, PerAxisWeighting) and
                      all(fac.ndim == 0 and np.isclose(fac, default_facs[i])
                          for i, fac in enumerate(self.weighting.factors))):
                    weighting_part = ''
                else:
                    weighting_part = self.weighting.repr_part

            optmod = [''] * len(optargs)
            if self.dtype.base in (float, complex, int, bool):
                optmod[0] = '!s'

            sep = ', ' if len(weighting_part) <= 30 else [', ', ',\n', ',\n']

            with npy_printoptions(precision=4):
                inner_str = signature_string(posargs, optargs, sep=sep,
                                             mod=[posmod, optmod])
            if len(weighting_part) <= 30:
                if weighting_part:
                    inner_str += ', ' + weighting_part
                return '{}({})'.format(ctor, inner_str)
            else:
                inner_str += ',\n' + weighting_part
                return '{}(\n{}\n)'.format(ctor, indent(inner_str))

        else:
            ctor = self.__class__.__name__
            posargs = [self.fspace, self.partition, self.tspace]
            optargs = [('interp', self.interp, 'nearest')]

            with npy_printoptions(precision=4):
                inner_str = signature_string(posargs, optargs,
                                             sep=[',\n', ', ', ',\n'],
                                             mod=['!r', '!s'])

            return '{}(\n{}\n)'.format(ctor, indent(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    @property
    def element_type(self):
        """`DiscreteLpElement`"""
        return DiscreteLpElement


class DiscreteLpElement(DiscretizedSpaceElement):

    """Representation of a `DiscreteLp` element."""

    # TODO: ndim_in, ndim_out, shape_in, shape_out

    @property
    def cell_sides(self):
        """Side lengths of a cell in an underlying *uniform* partition."""
        return self.space.cell_sides

    @property
    def cell_volume(self):
        """Cell volume of an underlying regular grid."""
        return self.space.cell_volume

    @property
    def data(self):
        """Data container of ``self``, depends on ``space.impl``."""
        return self.tensor.data

    @property
    def real(self):
        """Real part of this element.

        Returns
        -------
        real : `DiscreteLpElement`

        Examples
        --------
        Get the real part:

        >>> discr = uniform_discr(0, 1, 3, dtype=complex)
        >>> x = discr.element([5+1j, 3, 2-2j])
        >>> x.real
        uniform_discr(0.0, 1.0, 3).element([ 5.,  3.,  2.])

        Set the real part:

        >>> x = discr.element([1 + 1j, 2, 3 - 3j])
        >>> zero = discr.real_space.zero()
        >>> x.real = zero
        >>> x.real
        uniform_discr(0.0, 1.0, 3).element([ 0.,  0.,  0.])

        Other array-like types and broadcasting:

        >>> x.real = 1.0
        >>> x.real
        uniform_discr(0.0, 1.0, 3).element([ 1.,  1.,  1.])
        >>> x.real = [2, 3, 4]
        >>> x.real
        uniform_discr(0.0, 1.0, 3).element([ 2.,  3.,  4.])
        """
        return self.space.real_space.element(self.tensor.real)

    @real.setter
    def real(self, newreal):
        """Set the real part of this element to ``newreal``.

        This method is invoked by ``x.real = other``.

        Parameters
        ----------
        newreal : array-like or scalar
            Values to be assigned to the real part of this element.
        """
        self.tensor.real = newreal

    @property
    def imag(self):
        """Imaginary part of this element.

        Returns
        -------
        imag : `DiscreteLpElement`

        Examples
        --------
        Get the imaginary part:

        >>> discr = uniform_discr(0, 1, 3, dtype=complex)
        >>> x = discr.element([5+1j, 3, 2-2j])
        >>> x.imag
        uniform_discr(0.0, 1.0, 3).element([ 1.,  0., -2.])

        Set the imaginary part:

        >>> x = discr.element([1 + 1j, 2, 3 - 3j])
        >>> zero = discr.real_space.zero()
        >>> x.imag = zero
        >>> x.imag
        uniform_discr(0.0, 1.0, 3).element([ 0.,  0.,  0.])

        Other array-like types and broadcasting:

        >>> x.imag = 1.0
        >>> x.imag
        uniform_discr(0.0, 1.0, 3).element([ 1.,  1.,  1.])
        >>> x.imag = [2, 3, 4]
        >>> x.imag
        uniform_discr(0.0, 1.0, 3).element([ 2.,  3.,  4.])
        """
        return self.space.real_space.element(self.tensor.imag)

    @imag.setter
    def imag(self, newimag):
        """Set the imaginary part of this element to ``newimag``.

        This method is invoked by ``x.imag = other``.

        Parameters
        ----------
        newimag : array-like or scalar
            Values to be assigned to the imaginary part of this element.

        Raises
        ------
        ValueError
            If the space is real, i.e., no imagninary part can be set.
        """
        if self.space.is_real:
            raise ValueError('cannot set imaginary part in real spaces')
        self.tensor.imag = newimag

    def conj(self, out=None):
        """Complex conjugate of this element.

        Parameters
        ----------
        out : `DiscreteLpElement`, optional
            Element to which the complex conjugate is written.
            Must be an element of this element's space.

        Returns
        -------
        out : `DiscreteLpElement`
            The complex conjugate element. If ``out`` is provided,
            the returned object is a reference to it.

        Examples
        --------
        >>> discr = uniform_discr(0, 1, 4, dtype=complex)
        >>> x = discr.element([5+1j, 3, 2-2j, 1j])
        >>> y = x.conj()
        >>> print(y)
        [ 5.-1.j,  3.-0.j,  2.+2.j,  0.-1.j]

        The out parameter allows you to avoid a copy:

        >>> z = discr.element()
        >>> z_out = x.conj(out=z)
        >>> print(z)
        [ 5.-1.j,  3.-0.j,  2.+2.j,  0.-1.j]
        >>> z_out is z
        True

        It can also be used for in-place conjugation:

        >>> x_out = x.conj(out=x)
        >>> print(x)
        [ 5.-1.j,  3.-0.j,  2.+2.j,  0.-1.j]
        >>> x_out is x
        True
        """
        if out is None:
            return self.space.element(self.tensor.conj())
        else:
            self.tensor.conj(out=out.tensor)
            return out

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            The position(s) that should be accessed.

        Returns
        -------
        values : `DiscreteLpElement`
            The value(s) at the index (indices).

        Notes
        -----
        New axes can be created, however not within the axes that
        correspond to the function domain, only as new output axes.
        In other words, a scalar function can be turned into a
        ``(1, ..., 1)`` tensor-valued function this way by adding new axes
        of size 1, or a vector-valued function into a tensor-valued one
        with sizes 1 in the new axes. This is useful for broadcasting,
        see Examples.

        Examples
        --------
        # TODO: write example
        """
        if isinstance(indices, type(self)):
            indices = indices.tensor
        if isinstance(indices, type(self.tensor)):
            indices = indices.data
        if isinstance(indices, type(self.tensor.data)):
            return self.tensor[indices]

        try:
            iter(indices)
        except TypeError:
            pass

        # Normalize the index expression based on the full shape, and
        # split into "in" and "out" parts
        indices = normalized_index_expression(indices, self.shape)

        # New axes can be added to the output part, so we can only use
        # `indices_in` as fixed in length
        indices_in = indices[len(indices) - self.space.ndim_in:]
        # Now `indices_in` shouldn't contain any `None` since we don't
        # allow new axes there; we cannot use `indices.contains()` for
        # that check since it uses the `==` operator which will fail if
        # one of the indices is a Numpy array
        if any(idx is None for idx in indices_in):
            raise ValueError('new axes can be created only with respect to '
                             'the "output" dimensions, see the documentation')

        res_tens = self.tensor[indices]
        if np.isscalar(res_tens):
            return res_tens

        try:
            res_space = self.space[indices]
        except (ValueError, TypeError):
            # TODO: adapt error type
            # Cannot index space like this, e.g., with an index array or
            # a boolean array. Just return the tensor.
            return res_tens
        else:
            return res_space.element(res_tens)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Interface to Numpy's ufunc machinery.

        This method is called by Numpy version 1.13 and higher as a single
        point for the ufunc dispatch logic. An object implementing
        ``__array_ufunc__`` takes over control when a `numpy.ufunc` is
        called on it, allowing it to use custom implementations and
        output types.

        This includes handling of in-place arithmetic like
        ``npy_array += custom_obj``. In this case, the custom object's
        ``__array_ufunc__`` takes precedence over the baseline
        `numpy.ndarray` implementation. It will be called with
        ``npy_array`` as ``out`` argument, which ensures that the
        returned object is a Numpy array. For this to work properly,
        ``__array_ufunc__`` has to accept Numpy arrays as ``out`` arguments.

        See the `corresponding NEP`_ and the `interface documentation`_
        for further details. See also the `general documentation on
        Numpy ufuncs`_.

        .. note::
            When using operations that alter the shape (like ``reduce``),
            or the data type (can be any of the methods),
            the resulting array is wrapped in a space of the same
            type as ``self.space``, propagating all essential properties
            like weighting, exponent etc. as closely as possible.

        Parameters
        ----------
        ufunc : `numpy.ufunc`
            Ufunc that should be called on ``self``.
        method : str
            Method on ``ufunc`` that should be called on ``self``.
            Possible values:

            ``'__call__'``, ``'accumulate'``, ``'at'``, ``'outer'``,
            ``'reduce'``

        input1, ..., inputN :
            Positional arguments to ``ufunc.method``.
        kwargs :
            Keyword arguments to ``ufunc.method``.

        Returns
        -------
        ufunc_result : `DiscreteLpElement`, `numpy.ndarray` or tuple
            Result of the ufunc evaluation. If no ``out`` keyword argument
            was given, the result is a `DiscreteLpElement` or a tuple
            of such, depending on the number of outputs of ``ufunc``.
            If ``out`` was provided, the returned object or sequence members
            refer(s) to ``out``.

        Examples
        --------
        We apply `numpy.add` to elements of a one-dimensional space:

        >>> space = odl.uniform_discr(0, 1, 3)
        >>> x = space.element([1, 2, 3])
        >>> y = space.element([-1, -2, -3])
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        uniform_discr(0.0, 1.0, 3).element([ 0.,  0.,  0.])
        >>> np.add(x, y)  # same mechanism for Numpy >= 1.13
        uniform_discr(0.0, 1.0, 3).element([ 0.,  0.,  0.])

        As ``out``, a `DiscreteLpElement` can be provided as well as a
        `Tensor` of appropriate type, or its underlying data container
        type (wrapped in a sequence):

        >>> out = space.element()
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out,))
        >>> out
        uniform_discr(0.0, 1.0, 3).element([ 0.,  0.,  0.])
        >>> res is out
        True
        >>> out_tens = odl.rn(3).element()
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out_tens,))
        >>> out_tens
        rn(3).element([ 0.,  0.,  0.])
        >>> res is out_tens
        True
        >>> out_arr = np.empty(3)
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out_arr,))
        >>> out_arr
        array([ 0.,  0.,  0.])
        >>> res is out_arr
        True

        With multiple dimensions:

        >>> space_2d = odl.uniform_discr([0, 0], [1, 2], (2, 3))
        >>> x = y = space_2d.one()
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        uniform_discr([ 0.,  0.], [ 1.,  2.], (2, 3)).element(
            [[ 2.,  2.,  2.],
             [ 2.,  2.,  2.]]
        )

        The ``ufunc.accumulate`` method retains the original space:

        >>> x = space.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'accumulate', x)
        uniform_discr(0.0, 1.0, 3).element([ 1.,  3.,  6.])
        >>> np.add.accumulate(x)  # same mechanism for Numpy >= 1.13
        uniform_discr(0.0, 1.0, 3).element([ 1.,  3.,  6.])

        For multi-dimensional space elements, an optional ``axis`` parameter
        can be provided (default is 0):

        >>> z = space_2d.one()
        >>> z.__array_ufunc__(np.add, 'accumulate', z, axis=1)
        uniform_discr([ 0.,  0.], [ 1.,  2.], (2, 3)).element(
            [[ 1.,  2.,  3.],
             [ 1.,  2.,  3.]]
        )

        The method also takes a ``dtype`` parameter:

        >>> z.__array_ufunc__(np.add, 'accumulate', z, dtype=complex)
        uniform_discr([ 0.,  0.], [ 1.,  2.], (2, 3), dtype=complex).element(
            [[ 1.+0.j,  1.+0.j,  1.+0.j],
             [ 2.+0.j,  2.+0.j,  2.+0.j]]
        )

        The ``ufunc.at`` method operates in-place. Here we add the second
        operand ``[5, 10]`` to ``x`` at indices ``[0, 2]``:

        >>> x = space.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'at', x, [0, 2], [5, 10])
        >>> x
        uniform_discr(0.0, 1.0, 3).element([  6.,   2.,  13.])

        For outer-product-type operations, i.e., operations where the result
        shape is the sum of the individual shapes, the ``ufunc.outer``
        method can be used:

        >>> space1 = odl.uniform_discr(0, 1, 2)
        >>> space2 = odl.uniform_discr(0, 2, 3)
        >>> x = space1.element([0, 3])
        >>> y = space2.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'outer', x, y)
        uniform_discr([ 0.,  0.], [ 1.,  2.], (2, 3)).element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        >>> y.__array_ufunc__(np.add, 'outer', y, x)
        uniform_discr([ 0.,  0.], [ 2.,  1.], (3, 2)).element(
            [[ 1.,  4.],
             [ 2.,  5.],
             [ 3.,  6.]]
        )

        Using ``ufunc.reduce`` in 1D produces a scalar:

        >>> x = space.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'reduce', x)
        6.0

        In multiple dimensions, ``axis`` can be provided for reduction over
        selected axes:

        >>> z = space_2d.element([[1, 2, 3],
        ...                       [4, 5, 6]])
        >>> z.__array_ufunc__(np.add, 'reduce', z, axis=1)
        uniform_discr(0.0, 1.0, 2).element([  6.,  15.])

        References
        ----------
        .. _corresponding NEP:
           https://docs.scipy.org/doc/numpy/neps/ufunc-overrides.html

        .. _interface documentation:
           https://docs.scipy.org/doc/numpy/reference/arrays.classes.html\
#numpy.class.__array_ufunc__

        .. _general documentation on Numpy ufuncs:
           https://docs.scipy.org/doc/numpy/reference/ufuncs.html

        .. _reduceat documentation:
           https://docs.scipy.org/doc/numpy/reference/generated/\
        """
        # --- Process `out` --- #

        # Unwrap out if provided. The output parameters are all wrapped
        # in one tuple, even if there is only one.
        out_tuple = kwargs.pop('out', ())

        # Check number of `out` args, depending on `method`
        if method == '__call__' and len(out_tuple) not in (0, ufunc.nout):
            raise ValueError(
                "need 0 or {} `out` arguments for `method='__call__'`, "
                'got {}'.format(ufunc.nout, len(out_tuple)))
        elif method != '__call__' and len(out_tuple) not in (0, 1):
            raise ValueError(
                "need 0 or 1 `out` arguments for `method={!r}`, "
                'got {}'.format(method, len(out_tuple)))

        # We allow our own element type, tensors and their data containers
        # as `out`
        valid_out_types = (type(self),
                           type(self.tensor),
                           type(self.tensor.data))
        if not all(isinstance(o, valid_out_types) or o is None
                   for o in out_tuple):
            return NotImplemented

        # Assign to `out` or `out1` and `out2`, respectively (using the
        # `tensor` attribute if available)
        out = out1 = out2 = None
        if len(out_tuple) == 1:
            out = getattr(out_tuple[0], 'tensor', out_tuple[0])
        elif len(out_tuple) == 2:
            out1 = getattr(out_tuple[0], 'tensor', out_tuple[0])
            out2 = getattr(out_tuple[1], 'tensor', out_tuple[1])

        # --- Process `inputs` --- #

        # Pull out the `tensor` attributes from DiscreteLpElement instances
        # since we want to pass them to `self.tensor.__array_ufunc__`
        input_tensors = tuple(
            elem.tensor if isinstance(elem, type(self)) else elem
            for elem in inputs)

        # --- Get some parameters for later --- #

        # Need to filter for `keepdims` in case `method='reduce'` since it's
        # invalid (happening below)
        keepdims = kwargs.pop('keepdims', False)

        # Determine list of remaining axes from `axis` for `'reduce'`
        axis = kwargs.get('axis', None)
        if axis is None:
            reduced_axes = list(range(1, self.ndim))
        else:
            try:
                iter(axis)
            except TypeError:
                axis = (int(axis),)

            reduced_axes = [i for i in range(self.ndim) if i not in axis]

        # --- Evaluate ufunc --- #

        if method == '__call__':
            if ufunc.nout == 1:
                kwargs['out'] = (out,)
                res_tens = self.tensor.__array_ufunc__(
                    ufunc, '__call__', *input_tensors, **kwargs)

                if out is None:
                    # Wrap result tensor in appropriate DiscreteLp space.
                    # Make new function space based on result dtype,
                    # keep everything else, and get `tspace` from the result
                    # tensor.
                    out_dtype = (res_tens.dtype, self.space.fspace.out_shape)
                    fspace = FunctionSpace(self.space.fspace.domain,
                                           out_dtype)
                    res_space = DiscreteLp(
                        fspace, self.space.partition,
                        res_tens.space, self.space.interp_byaxis,
                        axis_labels=self.space.axis_labels)
                    result = res_space.element(res_tens)
                else:
                    result = out_tuple[0]

                return result

            elif ufunc.nout == 2:
                kwargs['out'] = (out1, out2)
                res1_tens, res2_tens = self.tensor.__array_ufunc__(
                    ufunc, '__call__', *input_tensors, **kwargs)

                if out1 is None:
                    # Wrap as for nout = 1
                    out_dtype = (res1_tens.dtype, self.space.fspace.out_shape)
                    fspace = FunctionSpace(self.space.fspace.domain,
                                           out_dtype)
                    res_space = DiscreteLp(
                        fspace, self.space.partition,
                        res1_tens.space, self.space.interp_byaxis,
                        axis_labels=self.space.axis_labels)
                    result1 = res_space.element(res1_tens)
                else:
                    result1 = out_tuple[0]

                if out2 is None:
                    # Wrap as for nout = 1
                    out_dtype = (res2_tens.dtype, self.space.fspace.out_shape)
                    fspace = FunctionSpace(self.space.fspace.domain,
                                           out_dtype)
                    res_space = DiscreteLp(
                        fspace, self.space.partition,
                        res2_tens.space, self.space.interp_byaxis,
                        axis_labels=self.space.axis_labels)
                    result2 = res_space.element(res2_tens)
                else:
                    result2 = out_tuple[1]

                return result1, result2

            else:
                raise NotImplementedError('nout = {} not supported'
                                          ''.format(ufunc.nout))

        elif method == 'reduce' and keepdims:
            raise ValueError(
                '`keepdims=True` cannot be used in `reduce` since there is '
                'no unique way to determine a function domain in collapsed '
                'axes')

        elif method == 'reduceat':
            # Makes no sense since there is no way to determine in which
            # space the result should live, except in special cases when
            # axes are being completely collapsed or don't change size.
            raise ValueError('`reduceat` not supported')

        elif (method == 'outer' and
              not all(isinstance(inp, type(self)) for inp in inputs)):
                raise TypeError(
                    "inputs must be of type {} for `method='outer'`, "
                    'got types {}'
                    ''.format(type(self), tuple(type(inp) for inp in inputs)))

        else:  # method != '__call__', and otherwise valid

            if method != 'at':
                # No kwargs allowed for 'at'
                kwargs['out'] = (out,)

            res_tens = self.tensor.__array_ufunc__(
                ufunc, method, *input_tensors, **kwargs)

            # Shortcut for scalar or no return value
            if np.isscalar(res_tens) or res_tens is None:
                # The first occurs for `reduce` with all axes,
                # the second for in-place stuff (`at` currently)
                return res_tens

            if out is None:
                # Wrap in appropriate DiscreteLp space depending on `method`
                if method == 'accumulate':
                    # Make `fspace` with appropriate dtype, get `tspace`
                    # from the result tensor and keep the rest
                    fspace = FunctionSpace(self.space.domain,
                                           out_dtype=res_tens.dtype)

                    res_space = DiscreteLp(
                        fspace, self.space.partition, res_tens.space,
                        self.space.interp_byaxis,
                        axis_labels=self.space.axis_labels)
                    result = res_space.element(res_tens)

                elif method == 'outer':
                    # Concatenate domains, partitions, interp, axis_labels,
                    # and determine `tspace` from the result tensor
                    inp1, inp2 = inputs
                    domain = inp1.space.domain.append(inp2.space.domain)
                    fspace = FunctionSpace(domain, out_dtype=res_tens.dtype)
                    part = inp1.space.partition.append(inp2.space.partition)
                    interp = (inp1.space.interp_byaxis +
                              inp2.space.interp_byaxis)
                    labels1 = [lbl + ' (1)' for lbl in inp1.space.axis_labels]
                    labels2 = [lbl + ' (2)' for lbl in inp2.space.axis_labels]
                    labels = labels1 + labels2

                    res_space = DiscreteLp(
                        fspace, part, res_tens.space, interp,
                        axis_labels=labels)
                    result = res_space.element(res_tens)

                elif method == 'reduce':
                    # Index space by axis using `reduced_axes`
                    res_space = self.space.byaxis_in[reduced_axes].astype(
                        res_tens.dtype)
                    result = res_space.element(res_tens)

                else:
                    raise RuntimeError('bad `method`')

            else:
                # `out` may be `out_tuple[0].tensor`, but we want to return
                # the original one
                result = out_tuple[0]

            return result

    def show(self, title=None, method='', coords=None, indices=None,
             force_show=False, fig=None, **kwargs):
        """Display the function graphically.

        Parameters
        ----------
        title : string, optional
            Set the title of the figure

        method : string, optional
            1d methods:

            - ``'plot'`` : graph plot (default for 1d data)
            - ``'scatter'`` : scattered 2d points (2nd axis <-> value)

            2d methods:

            - ``'imshow'`` : image plot with coloring according to value,
              including a colorbar (default for 2d data).
            - ``'scatter'`` : cloud of scattered 3d points
              (3rd axis <-> value)

        coords : `array-like`, optional
            Display a slice of the array instead of the full array.
            The values are shown accordinging to the given values,
            where ``None`` means all values along that dimension. For
            example, ``[None, None, 0.5]`` shows all values in the first
            two dimensions, with the third coordinate equal to 0.5.
            If a sequence is provided, it specifies the minimum and maximum
            point to be shown, i.e. ``[None, [0, 1]]`` shows all of the
            first axis and values between 0 and 1 in the second.
            This option is mutually exclusive with ``indices``.

        indices : int, slice, Ellipsis or sequence, optional
            Display a slice of the array instead of the full array.
            If a sequence is given, the i-th entry indexes the i-th axis,
            with the following behavior for the different types of entries:

                - ``int``: take entries with this index along axis ``i``,
                  removing this axis from the result
                - ``slice``: take a subset along axis ``i``, keeping it
                  intact
                - ``None``: equivalent to ``slice(None)``
                - ``Ellipsis`` (``...``): equivalent to the number of
                  ``None`` entries required to fill up the sequence to
                  correct length.

            The typical use case is to show a slice for a fixed index in
            a specific axis, which can be done most easily by setting, e.g.,
            ``indices=[None, 50, None]`` to take the 2d slice parallel to
            the x-z coordinate plane at index ``y = 50``.

            A single ``int`` or ``slice`` object indexes the first
            axis, i.e., is treated as ``(int_or_slice, Ellipsis)``.
            For the default ``None``, the array is kepty as-is for data
            that has at most 2 dimensions. For higher-dimensional
            data, the 2d slice in the first two axes at the middle
            position along the remaining axes is shown
            (semantically ``[:, :, shape[2:] // 2]``).
            This option is mutually exclusive with ``coords``.

        force_show : bool, optional
            Whether the plot should be forced to be shown now or deferred until
            later. Note that some backends always displays the plot, regardless
            of this value.

        fig : `matplotlib.figure.Figure`, optional
            The figure to show in. Expected to be of same "style", as
            the figure given by this function. The most common use case
            is that ``fig`` is the return value of an earlier call to
            this function.

        kwargs : {'figsize', 'saveto', 'clim', ...}, optional
            Extra keyword arguments passed on to the display method.
            See the Matplotlib functions for documentation of extra
            options.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The resulting figure. It is also shown to the user.

        See Also
        --------
        odl.util.graphics.show_discrete_data : Underlying implementation
        """
        from odl.util.graphics import show_discrete_data

        if 'interp' not in kwargs and not isinstance(self.space.interp, tuple):
            # Otherwise different interp per axis
            kwargs['interp'] = self.space.interp

        if self.ndim == 0:
            raise ValueError('nothing to show for 0-dimensional vector')

        if coords is not None:
            if indices is not None:
                raise ValueError('cannot provide both coords and indices')

            partition = self.space.partition
            shape = self.shape
            indices = []
            for axis, (n, coord) in enumerate(zip(shape, coords)):
                try:
                    coord_minp, coord_maxp = coord
                except TypeError:
                    coord_minp = coord_maxp = coord

                subpart = partition.byaxis[axis]

                # Validate input
                if coord_minp is not None:
                    coord_minp = subpart.set.element(coord_minp)
                if coord_maxp is not None:
                    coord_maxp = subpart.set.element(coord_maxp)

                if len(subpart) == 0:  # trivial cases
                    indices.append(0)
                elif coord_minp is not None and coord_minp == coord_maxp:
                    indices.append(subpart.index(coord_minp))
                else:
                    if coord_minp is None:
                        min_ind = 0
                    else:
                        min_ind = np.floor(subpart.index(coord_minp,
                                                         floating=True))

                    if coord_maxp is None:
                        max_ind = len(subpart)
                    else:
                        max_ind = np.ceil(subpart.index(coord_maxp,
                                                        floating=True))

                    indices.append(slice(int(min_ind), int(max_ind)))

        # Default to showing x-y slice "in the middle"
        if indices is None and self.ndim >= 3:
            indices = ((slice(None),) * 2 +
                       tuple(n // 2 for n in self.space.shape[2:]))

        # Normalize indices
        if is_int(indices) or isinstance(indices, slice):
            indices = (indices,)
        elif indices is None or indices == Ellipsis:
            indices = (slice(None),) * self.ndim

        # Single index or slice indexes the first axis, rest untouched
        if len(indices) == 1:
            indices = tuple(indices) + (Ellipsis,)

        # Convert `Ellipsis` objects
        if indices.count(Ellipsis) > 1:
            raise ValueError('cannot use more than 1 `Ellipsis` (`...`)')
        elif Ellipsis in indices:
            # Replace Ellipsis with the correct number of `slice(None)`
            pos = indices.index(Ellipsis)
            indices = (tuple(indices[:pos]) +
                       (slice(None),) * (self.ndim - len(indices) + 1) +
                       tuple(indices[pos + 1:]))

        # Now indices should be exactly of length `ndim`
        if len(indices) < self.ndim:
            raise ValueError('too few axes ({} < {})'.format(len(indices),
                                                             self.ndim))
        if len(indices) > self.ndim:
            raise ValueError('too many axes ({} > {})'.format(len(indices),
                                                              self.ndim))

        # Map `None` to `slice(None)` in indices for syntax like `coords`
        indices = tuple(slice(None) if idx is None else idx
                        for idx in indices)

        squeezed_axes = [axis for axis in range(self.ndim)
                         if is_int(indices[axis])]
        axis_labels = [self.space.axis_labels[axis] for axis in squeezed_axes]

        # Extend partition (trivially) to output axes
        part_out = uniform_partition_fromgrid(
            uniform_grid([0] * self.space.ndim_out, self.space.shape_out,
                         self.space.shape_out))
        part = part_out.append(self.space.partition)

        # Squeeze partition and values according to the index expression
        part = part[indices].squeeze()
        values = self.asarray()[indices].squeeze()

        return show_discrete_data(values, part, title=title, method=method,
                                  force_show=force_show, fig=fig,
                                  axis_labels=axis_labels, **kwargs)


def uniform_discr_frompartition(partition, dtype=None, impl='numpy', **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    partition : `RectPartition`
        Uniform partition to be used for discretization.
        It defines the domain and the functions and the grid for
        discretization.
    dtype : optional
        Data type of the created space. Can be provided in any way the
        `numpy.dtype` constructor understands, e.g. as built-in type or
        as a string.

        To create a space of vector- or tensor-valued functions,
        use a dtype with a shape, e.g., ``np.dtype((float, (2, 3)))``.
    impl : string, optional
        Implementation of the data storage arrays
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space.

    Examples
    --------
    >>> part = odl.uniform_partition(0, 1, 10)
    >>> uniform_discr_frompartition(part)
    uniform_discr(0.0, 1.0, 10)

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_fromspace : uniform Lp discretization from an existing
        function space
    odl.discr.partition.uniform_partition :
        partition of the function domain
    """
    if not isinstance(partition, RectPartition):
        raise TypeError('`partition` {!r} is not a `RectPartition` instance'
                        ''.format(partition))
    if not partition.is_uniform:
        raise ValueError('`partition` is not uniform')

    if dtype is not None:
        dtype = np.dtype(dtype)

    fspace = FunctionSpace(partition.set, out_dtype=dtype)
    ds_type = tspace_type(fspace, impl, fspace.scalar_out_dtype)

    if dtype is None:
        dtype = ds_type.default_dtype()

    weighting = kwargs.pop('weighting', None)
    exponent = kwargs.pop('exponent', 2.0)
    if (weighting is None and
            is_numeric_dtype(dtype) and
            exponent != float('inf')):
        weighting = np.concatenate([np.ones(len(fspace.out_shape)),
                                    partition.cell_sides])

    tspace = ds_type(fspace.out_shape + partition.shape, dtype.base,
                     exponent=exponent, weighting=weighting)
    return DiscreteLp(fspace, partition, tspace, **kwargs)


def uniform_discr_fromspace(fspace, shape, dtype=None, impl='numpy', **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    fspace : `FunctionSpace`
        Continuous function space. Its domain must be an `IntervalProd`.
    shape : int or sequence of ints
        Number of samples per axis.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : string, optional
        Implementation of the data storage arrays
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    >>> intv = odl.IntervalProd(0, 1)
    >>> fspace = odl.FunctionSpace(intv)
    >>> odl.uniform_discr_fromspace(fspace, 10)
    uniform_discr(0.0, 1.0, 10)

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromintv : uniform discretization from an existing
        interval product
    odl.discr.partition.uniform_partition :
        partition of the function domain
    """
    if not isinstance(fspace, FunctionSpace):
        raise TypeError('`fspace` {!r} is not a `FunctionSpace` instance'
                        ''.format(fspace))
    if not isinstance(fspace.domain, IntervalProd):
        raise TypeError('domain {!r} of the function space is not an '
                        '`IntervalProd` instance'.format(fspace.domain))

    # Set data type. If given, check consistency with fspace's field and
    # out_dtype. If not given, take the latter.
    if dtype is None:
        dtype = fspace.out_dtype
    else:
        dtype, dtype_in = np.dtype(dtype), dtype
        if not np.can_cast(fspace.scalar_out_dtype, dtype, casting='safe'):
            raise ValueError('cannot safely cast from output data {} type of '
                             'the function space to given data type {}'
                             ''.format(fspace.out, dtype_in))
        if dtype.shape != fspace.out_shape:
            raise ValueError('`dtype.shape` {} not equal to '
                             '`fspace.out_shape` {}'
                             ''.format(dtype.shape, fspace.out_shape))

    if fspace.field == RealNumbers() and not is_real_dtype(dtype):
        raise ValueError('cannot discretize real space {} with '
                         'non-real data type {}'
                         ''.format(fspace, dtype))
    elif (fspace.field == ComplexNumbers() and
          not is_complex_floating_dtype(dtype)):
        raise ValueError('cannot discretize complex space {} with '
                         'non-complex-floating data type {}'
                         ''.format(fspace, dtype))

    nodes_on_bdry = kwargs.pop('nodes_on_bdry', False)
    partition = uniform_partition_fromintv(fspace.domain, shape, nodes_on_bdry)
    return uniform_discr_frompartition(partition, dtype, impl, **kwargs)


def uniform_discr_fromintv(intv_prod, shape, dtype=None, impl='numpy',
                           **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    intv_prod : `IntervalProd`
        Function domain of the uniformly discretized space.
    shape : int or sequence of ints
        Number of samples per axis.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : str, optional
        Implementation of the data storage arrays.
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    >>> intv = IntervalProd(0, 1)
    >>> uniform_discr_fromintv(intv, 10)
    uniform_discr(0.0, 1.0, 10)

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromspace : uniform discretization from an existing
        function space
    """
    if dtype is None:
        dtype = tensor_space_impl(str(impl).lower()).default_dtype()

    fspace = FunctionSpace(intv_prod, out_dtype=dtype)
    return uniform_discr_fromspace(fspace, shape, dtype, impl, **kwargs)


def uniform_discr(min_pt, max_pt, shape, dtype=None, impl='numpy', **kwargs):
    """Return a uniformly discretized L^p function space.

    Parameters
    ----------
    min_pt, max_pt : float or sequence of floats
        Minimum/maximum corners of the desired function domain.
    shape : int or sequence of ints
        Number of samples per axis.
    dtype : optional
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : string, optional
        Implementation of the data storage arrays.

    Other Parameters
    ----------------
    exponent : positive float, optional
        The parameter :math:`p` in :math:`L^p`. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : string or sequence of strings, optional
        Interpolation type to be used for discretization.
        A sequence is interpreted as interpolation scheme per axis.
        Possible values:

        - ``'nearest'`` : use nearest-neighbor interpolation.
        - ``'linear'`` : use linear interpolation.

    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``len(shape)``.
        A single boolean is interpreted as a global choice for all
        boundaries.
        Default: ``False``.
    weighting : optional
        Use weighted inner product, norm, and dist. The following
        types are supported as ``weighting``:

        - ``None``: Use the cell sides as weighting constants (default).
        - ``float``: Weighting by a constant.
        - sequence of length ``ndim``: Separate weighting per axis.
          Entries can be constants or 1D arrays.
        - array-like: Pointwise weighting by an array of the same
          ``shape`` as this space.
        - `Weighting`: Use this weighting as-is. Compatibility
          with this space is not checked during init.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    Create real space:

    >>> space = uniform_discr([0, 0], [1, 1], (10, 10))
    >>> space
    uniform_discr([ 0.,  0.], [ 1.,  1.], (10, 10))
    >>> space.cell_sides
    array([ 0.1,  0.1])
    >>> space.dtype
    dtype('float64')
    >>> space.is_real
    True

    Create complex space by giving a dtype:

    >>> space = uniform_discr([0, 0], [1, 1], (10, 10), dtype=complex)
    >>> space
    uniform_discr([ 0.,  0.], [ 1.,  1.], (10, 10), dtype=complex)
    >>> space.is_complex
    True
    >>> space.real_space  # Get real counterpart
    uniform_discr([ 0.,  0.], [ 1.,  1.], (10, 10))

    See Also
    --------
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromspace : uniform discretization from an existing
        function space
    uniform_discr_fromintv : uniform discretization from an existing
        interval product
    """
    intv_prod = IntervalProd(min_pt, max_pt)
    return uniform_discr_fromintv(intv_prod, shape, dtype, impl, **kwargs)


def uniform_discr_fromdiscr(discr, min_pt=None, max_pt=None,
                            shape=None, cell_sides=None, **kwargs):
    """Return a discretization based on an existing one.

    The parameters that are explicitly given are used to create the
    new discretization, and the missing information is taken from
    the template space. See Notes for the exact functionality.

    Parameters
    ----------
    discr : `DiscreteLp`
        Uniformly discretized space used as a template.
    min_pt, max_pt: float or sequence of floats, optional
        Desired minimum/maximum corners of the new space domain.
    shape : int or sequence of ints, optional
        Desired number of samples per axis of the new space.
    cell_sides : float or sequence of floats, optional
        Desired cell side lenghts of the new space's partition.
    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``discr.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.

        Default: ``False``.

    kwargs :
        Additional keyword parameters passed to the `DiscreteLp`
        initializer.

    Notes
    -----
    The parameters ``min_pt``, ``max_pt``, ``shape`` and
    ``cell_sides`` can be combined in the following ways (applies in
    each axis individually):

    **0 arguments:**
        Return a copy of ``discr``

    **1 argument:**
        [min,max]_pt -> keep sampling but translate domain so it
        starts/ends at ``[min,max]_pt``

        shape/cell_sides -> keep domain but change sampling.
        See `uniform_partition` for restrictions.

    **2 arguments:**
        min_pt + max_pt -> translate and resample with the same
        number of samples

        [min,max]_pt + shape/cell_sides -> translate and resample

        shape + cell_sides -> error due to ambiguity (keep
        ``min_pt`` or ``max_pt``?)

    **3+ arguments:**
        The underlying partition is uniquely determined by the new
        parameters. See `uniform_partition`.

    See Also
    --------
    uniform_discr : implicit uniform Lp discretization
    odl.discr.partition.uniform_partition :
        underlying domain partitioning scheme

    Examples
    --------
    >>> discr = odl.uniform_discr([0, 0], [1, 2], (10, 5))
    >>> discr.cell_sides
    array([ 0.1,  0.4])

    If no additional argument is given, a copy of ``discr`` is
    returned:

    >>> odl.uniform_discr_fromdiscr(discr) == discr
    True
    >>> odl.uniform_discr_fromdiscr(discr) is discr
    False

    Giving ``min_pt`` or ``max_pt`` results in a
    translation, while for the other two options, the domain
    is kept but re-partitioned:

    >>> odl.uniform_discr_fromdiscr(discr, min_pt=[1, 1])
    uniform_discr([ 1.,  1.], [ 2.,  3.], (10, 5))
    >>> odl.uniform_discr_fromdiscr(discr, max_pt=[0, 0])
    uniform_discr([-1., -2.], [ 0.,  0.], (10, 5))
    >>> odl.uniform_discr_fromdiscr(discr, cell_sides=[1, 1])
    uniform_discr([ 0.,  0.], [ 1.,  2.], (1, 2))
    >>> odl.uniform_discr_fromdiscr(discr, shape=[5, 5])
    uniform_discr([ 0.,  0.], [ 1.,  2.], (5, 5))
    >>> odl.uniform_discr_fromdiscr(discr, shape=[5, 5]).cell_sides
    array([ 0.2,  0.4])

    The cases with 2 or more additional arguments and the syntax
    for specifying quantities per axis is illustrated in the following:

    # axis 0: translate to match max_pt = 3
    # axis 1: recompute max_pt using the original shape with the
    # new min_pt and cell_sides
    >>> new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=[None, 1],
    ...                                         max_pt=[3, None],
    ...                                         cell_sides=[None, 0.25])
    >>> new_discr
    uniform_discr([ 2.,  1.], [ 3.  ,  2.25], (10, 5))
    >>> new_discr.cell_sides
    array([ 0.1 ,  0.25])

    # axis 0: recompute min_pt from old cell_sides and new
    # max_pt and shape
    # axis 1: use new min_pt, shape and cell_sides only
    >>> new_discr = odl.uniform_discr_fromdiscr(discr, min_pt=[None, 1],
    ...                                         max_pt=[3, None],
    ...                                         shape=[5, 5],
    ...                                         cell_sides=[None, 0.25])
    >>> new_discr
    uniform_discr([ 2.5,  1. ], [ 3.  ,  2.25], (5, 5))
    >>> new_discr.cell_sides
    array([ 0.1 ,  0.25])
    """
    if not isinstance(discr, DiscreteLp):
        raise TypeError('`discr` {!r} is not a DiscreteLp instance'
                        ''.format(discr))
    if not discr.is_uniform:
        raise ValueError('`discr` {} is not uniformly discretized'
                         ''.format(discr))

    # Normalize partition parameters
    min_pt = normalized_scalar_param_list(min_pt, discr.ndim,
                                          param_conv=float, keep_none=True)
    max_pt = normalized_scalar_param_list(max_pt, discr.ndim,
                                          param_conv=float, keep_none=True)
    shape = normalized_scalar_param_list(shape, discr.ndim,
                                         param_conv=safe_int_conv,
                                         keep_none=True)
    cell_sides = normalized_scalar_param_list(cell_sides, discr.ndim,
                                              param_conv=float, keep_none=True)

    nodes_on_bdry = kwargs.pop('nodes_on_bdry', False)
    nodes_on_bdry = normalized_nodes_on_bdry(nodes_on_bdry, discr.ndim)

    new_min_pt = []
    new_max_pt = []
    new_shape = []
    new_csides = []
    for i, (xmin, xmax, n, s, old_xmin, old_xmax, old_n, old_s) in enumerate(
            zip(min_pt, max_pt, shape, cell_sides,
                discr.min_pt, discr.max_pt, discr.shape,
                discr.cell_sides)):
        num_params = sum(p is not None for p in (xmin, xmax, n, s))

        if num_params == 0:
            new_params = [old_xmin, old_xmax, old_n, None]

        elif num_params == 1:
            if xmin is not None:
                new_params = [xmin, old_xmax + (xmin - old_xmin), old_n, None]
            elif xmax is not None:
                new_params = [old_xmin + (xmax - old_xmax), xmax, old_n, None]
            elif n is not None:
                new_params = [old_xmin, old_xmax, n, None]
            else:
                new_params = [old_xmin, old_xmax, None, s]

        elif num_params == 2:
            if xmin is not None and xmax is not None:
                new_params = [xmin, xmax, old_n, None]
            elif xmin is not None and n is not None:
                new_params = [xmin, None, n, old_s]
            elif xmin is not None and s is not None:
                new_params = [xmin, None, old_n, s]
            elif xmax is not None and n is not None:
                new_params = [None, xmax, n, old_s]
            elif xmax is not None and s is not None:
                new_params = [None, xmax, old_n, s]
            else:
                raise ValueError('in axis {}: cannot use `shape` and '
                                 '`cell_size` only due to ambiguous values '
                                 'for `min_pt` and `max_pt`.'.format(i))

        else:
            new_params = [xmin, xmax, n, s]

        new_min_pt.append(new_params[0])
        new_max_pt.append(new_params[1])
        new_shape.append(new_params[2])
        new_csides.append(new_params[3])

    new_part = uniform_partition(min_pt=new_min_pt, max_pt=new_max_pt,
                                 shape=new_shape,
                                 cell_sides=new_csides,
                                 nodes_on_bdry=nodes_on_bdry)

    return uniform_discr_frompartition(new_part, exponent=discr.exponent,
                                       interp=discr.interp, impl=discr.impl,
                                       **kwargs)


def _scaling_func_list(bdry_fracs, exponent):
    """Return a list of lists of scaling functions for the boundary."""
    def scaling(factor):
        def scaling_func(x):
            return x * factor
        return scaling_func

    func_list = []
    for frac_l, frac_r in bdry_fracs:
        func_list_entry = []
        if np.isclose(frac_l, 1.0):
            func_list_entry.append(None)
        else:
            func_list_entry.append(scaling(frac_l ** (1 / exponent)))

        if np.isclose(frac_r, 1.0):
            func_list_entry.append(None)
        else:
            func_list_entry.append(scaling(frac_r ** (1 / exponent)))

        func_list.append(func_list_entry)
    return func_list


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
