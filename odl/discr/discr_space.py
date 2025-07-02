# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Lebesgue L^p type discretizations of function spaces."""

from __future__ import absolute_import, division, print_function

from numbers import Integral

import numpy as np

from odl.discr.discr_utils import point_collocation, sampling_function
from odl.discr.partition import (
    RectPartition, uniform_partition, uniform_partition_fromintv)
from odl.set import IntervalProd, RealNumbers
from odl.set.space import LinearSpace, SupportedNumOperationParadigms, NumOperationParadigmSupport
from odl.space import ProductSpace
from odl.space.base_tensors import Tensor, TensorSpace, default_dtype
from odl.space.entry_points import tensor_space_impl
from odl.space.weighting import ConstWeighting
from odl.util import (
    apply_on_boundary, array_str, dtype_str, is_floating_dtype,
    is_numeric_dtype, normalized_nodes_on_bdry, normalized_scalar_param_list,
    repr_string, safe_int_conv, signature_string_parts)

__all__ = (
    'DiscretizedSpace',
    'DiscretizedSpaceElement',
    'uniform_discr_frompartition',
    'uniform_discr_fromintv',
    'uniform_discr',
    'uniform_discr_fromdiscr',
)


class DiscretizedSpace(TensorSpace):

    """Discretization of a Lebesgue :math:`L^p` space."""

    def __init__(self, partition, tspace, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        partition : `RectPartition`
            Partition of a rectangular spatial domain.
        tspace : `TensorSpace`
            Space of elements used for data storage. It must have the same
            `TensorSpace.shape` as ``partition``.
        axis_labels : sequence of str, optional
            Names of the axes to use for plotting etc.
            Default:

            - 1D: ``['$x$']``
            - 2D: ``['$x$', '$y$']``
            - 3D: ``['$x$', '$y$', '$z$']``
            - nD: ``['$x_1$', '$x_2$', ..., '$x_n$']``

            Note: The ``$`` signs ensure rendering as LaTeX.
        """
        if not isinstance(partition, RectPartition):
            raise TypeError('`partition` must be a `RectPartition`, got {!r}'
                            ''.format(partition))
        if not isinstance(tspace, TensorSpace):
            raise TypeError('`tspace` must be a `TensorSpace`, got {!r}'
                            ''.format(tspace))
        if partition.shape != tspace.shape:
            raise ValueError(
                '`partition.shape` must be equal to `tspace.shape`, but '
                '{} != {}'.format(partition.shape, tspace.shape)
            )

        self.__tspace = tspace
        self.__partition = partition

        self._init_dtype(tspace.dtype)

        self._init_shape(tspace.shape, tspace.dtype)

        self._init_device(tspace.device)

        self.__use_in_place_ops = kwargs.pop('use_in_place_ops', True)
  
        self._init_weighting()

        field = self._init_field()

        LinearSpace.__init__(self, field)

        # Set axis labels
        axis_labels = kwargs.pop('axis_labels', None)
        if axis_labels is None:
            if self.ndim <= 3:
                self.__axis_labels = ('$x$', '$y$', '$z$')[:self.ndim]
            else:
                self.__axis_labels = tuple('$x_{}$'.format(axis)
                                           for axis in range(self.ndim))
        else:
            self.__axis_labels = tuple(str(label) for label in axis_labels)

        if kwargs:
            raise ValueError('got unexpected keyword arguments {}'
                             ''.format(kwargs))

    # --- Meta-info

    @property
    def element_type(self):
        """`DiscretizedSpaceElement`"""
        return DiscretizedSpaceElement

    @property
    def supported_num_operation_paradigms(self) -> NumOperationParadigmSupport:
        """In-place vs out-of-place is not of much concern for the discretization
        and only depends on the underlying arrays."""
        return self.tspace.supported_num_operation_paradigms

    # --- Constructor args

    @property
    def partition(self):
        """`RectPartition` of the function domain."""
        return self.__partition

    @property
    def tspace(self):
        """Space for the coefficients of the elements of this space."""
        return self.__tspace

    @property
    def axis_labels(self):
        """Labels for axes when displaying space elements."""
        return self.__axis_labels

    # --- Pass-through `partition` attributes

    @property
    def domain(self):
        """Set on which functions are defined before discretization."""
        return self.partition.set

    # --- Pass-through `tspace` attributes

    @property
    def weighting(self):
        """This space's weighting scheme."""
        # TODO(kohr-h): `weighting` is optional in `tspace`, how should we
        # handle that?
        return self.tspace.weighting

    @property
    def is_weighted(self):
        """``True`` if the ``tspace`` is weighted."""
        return getattr(self.tspace, 'is_weighted', False)

    @property
    def impl(self):
        """Name of the implementation back-end."""
        return self.tspace.impl

    @property
    def exponent(self):
        """Exponent of this space, the ``p`` in ``L^p``."""
        # TODO(kohr-h): `exponent` is optional in `tspace`, how should we
        # handle that?
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
        """Shape of the underlying partition."""
        return self.partition.shape

    @property
    def ndim(self):
        """Number of dimensions (= number of axes)."""
        return self.partition.ndim

    @property
    def size(self):
        """Total number of underlying partition cells."""
        return self.partition.size

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

    def points(self):
        """All sampling points in the partition.

        Returns
        -------
        points : `numpy.ndarray`
            The shape of the array is ``size x ndim``, i.e. the points
            are stored as rows.
        """
        return self.partition.points()

    def available_dtypes(self):
        """Available data types for new elements in this space.

        This is equal to the available data types of `tspace`.
        """
        return self.tspace.available_dtypes()

    # --- Derived properties

    @property
    def tspace_type(self):
        """Tensor space type of this space."""
        return type(self.tspace)

    @property
    def tangent_bundle(self):
        """The tangent bundle associated with `domain` using `partition`.

        The tangent bundle of a space ``X`` of functions ``R^d --> F`` can be
        interpreted as the space of vector-valued functions ``R^d --> F^d``.
        This space can be identified with the power space ``X^d`` as used
        in this implementation.
        """
        if self.ndim == 0:
            return ProductSpace(field=self.field)
        else:
            return ProductSpace(self, self.ndim)

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

    # --- Element creation

    def element(self, inp=None, **kwargs):
        """Create an element from ``inp`` or from scratch.

        Parameters
        ----------
        inp : optional
            Input used to initialize the new element. The following options
            are available:

            - ``None``: an empty element is created with no guarantee of
              its state (memory allocation only).

            - array-like: an element wrapping a `tensor` is created,
              where a copy is avoided whenever possible. This usually
              requires correct `shape`, `dtype` and `impl` if applicable.
              See the ``element`` method of `tspace` for more
              information.

              If any of these conditions is not met, a copy is made.

            - callable: a new element is created by sampling the function
              using `point_collocation`.

        kwargs :
            Additional arguments passed on to `point_collocation` when
            called on ``inp``, in the form
            ``point_collocation(inp, points, **kwargs)``.
            This can be used e.g. for functions with parameters.

        Returns
        -------
        element : `DiscretizedSpaceElement`
            The discretized element, calculated as ``point_collocation(inp)``
            or ``tspace.element(inp)``, tried in this order.

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
        """
        if inp is None:
            return self.element_type(self, self.tspace.element())
        elif inp in self:
            return inp
        elif inp in self.tspace:
            return self.element_type(self, inp)
        elif callable(inp):
            func = sampling_function(
                inp, self.domain, out_dtype=self.dtype,
            )
            sampled = point_collocation(func, self.meshgrid, **kwargs)
            return self.element_type(
                self, self.tspace.element(sampled)
            )
        else:
            # Sequence-type input
            return self.element_type(
                self, self.tspace.element(inp)
            )

    def zero(self):
        """Return the element of all zeros."""
        return self.element_type(self, self.tspace.zero())

    def one(self):
        """Return the element of all ones."""
        return self.element_type(self, self.tspace.one())

    # --- Casting

    def _astype(self, dtype):
        """Internal helper for ``astype``."""
        tspace = self.tspace.astype(dtype)
        return type(self)(
            self.partition, tspace, axis_labels=self.axis_labels)

    # --- Slicing

    # TODO: add `byaxis`_out when discretized tensor-valued functions are
    # available

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

        class DiscretizedSpaceByaxisIn(object):

            """Helper class for indexing by domain axes."""

            def __getitem__(self, indices):
                """Return ``self[indices]``.

                Parameters
                ----------
                indices : index expression
                    Object used to index the space domain.

                Returns
                -------
                space : `DiscretizedSpace`
                    The resulting space with indexed domain and otherwise
                    same properties (except possibly weighting).
                """
                part = space.partition.byaxis[indices]

                if isinstance(space.weighting, ConstWeighting):
                    # Need to manually construct `tspace` since it doesn't
                    # know where its weighting factor comes from
                    try:
                        iter(indices)
                    except TypeError:
                        newshape = space.shape[indices]
                    else:
                        newshape = tuple(space.shape[int(i)] for i in indices)

                    weighting = part.cell_volume
                    tspace = type(space.tspace)(
                        newshape, space.dtype,
                        exponent=space.exponent, weighting=weighting)
                else:
                    # Other weighting schemes are handled correctly by
                    # the tensor space
                    # TODO(kohr-h): `byaxis` is not guaranteed to exist in
                    # `tspace`, how to handle that?
                    tspace = space.tspace.byaxis[indices]

                try:
                    iter(indices)
                except TypeError:
                    labels = space.axis_labels[indices]
                else:
                    labels = tuple(space.axis_labels[int(i)]
                                   for i in indices)

                return DiscretizedSpace(part, tspace, axis_labels=labels)

            def __repr__(self):
                """Return ``repr(self)``."""
                return repr(space) + '.byaxis_in'

        return DiscretizedSpaceByaxisIn()

    # --- Identity

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if ``other`` is a `DiscretizedSpace` with equal
            `tspace`, ``False`` otherwise.
        """
        # Optimizations for simple cases
        if other is self:
            return True
        elif other is None:
            return False
        else:
            return (
                super(DiscretizedSpace, self).__eq__(other)
                and other.tspace == self.tspace
                and other.partition == self.partition
            )

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash(
            (super(DiscretizedSpace, self).__hash__(),
             self.tspace,
             self.partition)
        )

    # --- Space functions

    def _lincomb(self, a, x1, b, x2, out):
        """Raw linear combination."""
        return self.element(
            self.tspace._lincomb(a, x1.tensor, b, x2.tensor,
                                 out.tensor if out is not None else None))

    def _multiply(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        return self.element(
                self.tspace._multiply(x1.tensor, x2.tensor,
                                      out.tensor if out is not None else None))

    def _divide(self, x1, x2, out):
        """Raw pointwise multiplication of two elements."""
        return self.element(
                self.tspace._divide(x1.tensor, x2.tensor,
                                    out.tensor if out is not None else None))

    # The inherited methods by default use a weighting by a constant
    # (the grid cell size). In dimensions where the partitioned set contains
    # only a fraction of the outermost cells (e.g. if the outermost grid
    # points lie at the boundary), the corresponding contributions to
    # discretized integrals need to be scaled by that fraction.
    def _inner(self, x, y):
        """Return ``self.inner(x, y)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            # TODO: implement without copying x
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=1.0)
            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return self.tspace.inner(self.tspace.element(x_arr), y.tensor)
        else:
            return self.tspace.inner(x.tensor, y.tensor)

    def _norm(self, x):
        """Return ``self.norm(x)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            # TODO: implement without copying x
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)
            x_arr = apply_on_boundary(x, func=func_list, only_once=False)
            return self.tspace.norm(self.tspace.element(x_arr))
        else:
            return self.tspace.norm(x.tensor)

    def _dist(self, x, y):
        """Return ``self.dist(x, y)``."""
        if self.is_uniform and not self.is_uniformly_weighted:
            bdry_fracs = self.partition.boundary_cell_fractions
            func_list = _scaling_func_list(bdry_fracs, exponent=self.exponent)
            arrs = [apply_on_boundary(vec, func=func_list, only_once=False)
                    for vec in (x, y)]

            return self.tspace.dist(
                self.tspace.element(arrs[0]),
                self.tspace.element(arrs[1]),
            )
        else:
            return self.tspace.dist(x.tensor, y.tensor)

    def __repr__(self):
        """Return ``repr(self)``."""
        # Clunky check if the factory repr can be used
        if uniform_partition_fromintv(
            self.partition.set, self.shape, nodes_on_bdry=False
        ) == self.partition:
            use_uniform = True
            nodes_on_bdry = False
        elif uniform_partition_fromintv(
            self.partition.set, self.shape, nodes_on_bdry=True
        ) == self.partition:
            use_uniform = True
            nodes_on_bdry = True
        else:
            use_uniform = False
            nodes_on_bdry = None

        if use_uniform:
            ctor = 'uniform_discr'
            if self.ndim == 1:
                posargs = [self.min_pt[0], self.max_pt[0], self.shape[0]]
                posmod = ['', '', '']
            else:
                posargs = [self.min_pt, self.max_pt, self.shape]
                posmod = [array_str, array_str, '']

            default_dtype_s = dtype_str(
                default_dtype(self.tspace.array_backend, RealNumbers())
            )

            dtype_s = dtype_str(self.dtype)
            optargs = [
                ('impl', self.impl, 'numpy'),
                ('nodes_on_bdry', nodes_on_bdry, False),
                ('dtype', dtype_s, default_dtype_s)
            ]

            # Add weighting stuff if not equal to default
            if (
                self.exponent == float('inf')
                or self.ndim == 0
                or not is_floating_dtype(self.dtype)
            ):
                # In these cases, weighting constant 1 is the default
                if (
                    not isinstance(self.weighting, ConstWeighting)
                    or not np.isclose(self.weighting.const, 1.0)
                ):
                    optargs.append(('weighting', self.weighting.const, None))
            else:
                if (
                    not isinstance(self.weighting, ConstWeighting)
                    or not np.isclose(self.weighting.const, self.cell_volume)
                ):
                    optargs.append(('weighting', self.weighting.const, None))

            optmod = [''] * len(optargs)
            if self.dtype in (float, complex, int, bool):
                optmod[2] = '!s'

            inner_parts = signature_string_parts(
                posargs, optargs, [posmod, optmod]
            )
            return repr_string(ctor, inner_parts)

        else:
            ctor = self.__class__.__name__
            posargs = [self.partition, self.tspace]
            inner_parts = signature_string_parts(posargs, [])
            return repr_string(ctor, inner_parts, allow_mixed_seps=False)
        
    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)


class DiscretizedSpaceElement(Tensor):

    """Representation of a `DiscretizedSpace` element."""

    def __init__(self, space, tensor):
        """Initialize a new instance."""
        super(DiscretizedSpaceElement, self).__init__(space)
        self.__tensor = tensor

    # --- Constructor args

    @property
    def tensor(self):
        """Structure for data storage."""
        return self.__tensor

    # --- Pass-through `space` properties

    @property
    def cell_sides(self):
        """Side lengths of a cell in an underlying *uniform* partition."""
        return self.space.cell_sides

    @property
    def cell_volume(self):
        """Cell volume of an underlying regular grid."""
        return self.space.cell_volume

    # --- Pass-through `tensor` properties

    @property
    def data(self):
        """Data container of ``self``, depends on ``space.impl``."""
        return self.tensor.data

    @property
    def dtype(self):
        """Type of data storage."""
        return self.tensor.dtype

    @property
    def size(self):
        """Size of data storage."""
        return self.tensor.size

    def __len__(self):
        """Return ``len(self)``.

        Equivalent to ``self.shape[0]`` if possible. Zero-dimensional
        tensors have no length and produce a `TypeError`.
        """
        return len(self.tensor)

    def copy(self):
        """Create an identical (deep) copy of this element."""
        return self.space.element(self.tensor.copy())

    def asarray(self, out=None, must_be_contiguous=False):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.
        """
        return self.tensor.asarray(out=out, must_be_contiguous=must_be_contiguous)

    def astype(self, dtype):
        """Return a copy of this element with new ``dtype``.

        Parameters
        ----------
        dtype :
            Scalar data type of the returned space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.

        Returns
        -------
        newelem : `DisceteLpElement`
            Version of this element with given data type.
        """
        return self.space.astype(dtype).element(self.tensor.astype(dtype))

    def _assign(self, other, avoid_deep_copy):
        """Assign the values of ``other``, which is assumed to be in the
        same discretized space, to ``self``."""
        self.__tensor.assign(other.tensor, avoid_deep_copy=avoid_deep_copy)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            ``True`` if all entries of ``other`` are equal to this
            element's entries, ``False`` otherwise.
        """
        return other in self.space and other.tensor == self.tensor

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be accessed.

        Returns
        -------
        values : `Tensor`
            The value(s) at the index (indices).
        """
        if isinstance(indices, type(self)):
            indices = indices.tensor
        return self.tensor[indices]

    def __ipow__(self, p):
        """Implement ``self **= p``."""
        # The concrete `tensor` can specialize `__ipow__` for non-integer
        # `p` so we want to use it here. Otherwise we get the default
        # `LinearSpaceElement.__ipow__` which only works for integer `p`.
        self.tensor.__ipow__(p)
        return self

    @property
    def real(self):
        """Real part of this element.

        Returns
        -------
        real : `DiscretizedSpaceElement`

        Examples
        --------
        Get the real part:

        >>> discr = odl.uniform_discr(0, 1, 3, dtype=complex)
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
        if isinstance(newreal, DiscretizedSpaceElement):
            self.tensor.real = newreal.tensor
        else:
            self.tensor.real = newreal

    @property
    def imag(self):
        """Imaginary part of this element.

        Returns
        -------
        imag : `DiscretizedSpaceElement`

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
        if isinstance(newimag, DiscretizedSpaceElement):
            self.tensor.imag = newimag.tensor
        else:
            self.tensor.imag = newimag

    def conj(self, out=None):
        """Complex conjugate of this element.

        Parameters
        ----------
        out : `DiscretizedSpaceElement`, optional
            Element to which the complex conjugate is written.
            Must be an element of this element's space.

        Returns
        -------
        out : `DiscretizedSpaceElement`
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

    def __setitem__(self, indices, values):
        """Set values of this element.

        Parameters
        ----------
        indices : int or `slice`
            The position(s) that should be set
        values : scalar or `array-like`
            Value(s) to be assigned.
            If ``indices`` is an integer, ``values`` must be a scalar
            value.
            If ``indices`` is a slice, ``values`` must be
            broadcastable to the size of the slice (same size,
            shape ``(1,)`` or scalar).
            For ``indices == slice(None)``, i.e. in the call
            ``vec[:] = values``, a multi-dimensional array of correct
            shape is allowed as ``values``.
        """
        if values in self.space:
            self.tensor[indices] = values.tensor
        else:
            if isinstance(indices, type(self)):
                indices = indices.tensor
            if isinstance(values, type(self)):
                values = values.tensor
            self.tensor.__setitem__(indices, values)


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

        Other Parameters
        ----------------
        interp : {'linear', 'nearest'}, optional
            Interpolation type that should be used for the plot.

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

        if 'interp' not in kwargs:
            kwargs['interp'] = 'linear'

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
        if isinstance(indices, (Integral, slice)):
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
                         if not isinstance(indices[axis], Integral)]
        axis_labels = [self.space.axis_labels[axis] for axis in squeezed_axes]

        # Squeeze grid and values according to the index expression
        part = self.space.partition[indices].squeeze()
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
        Data type for the discretized space, must be understood by the
        `numpy.dtype` constructor. The default for ``None`` depends on the
        ``impl`` backend, usually it is ``'float64'`` or ``'float32'``.
    impl : string, optional
        Implementation of the data storage arrays
    kwargs :
        Additional keyword parameters, see `uniform_discr` for details.

    Returns
    -------
    discr : `DiscretizedSpace`
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

    tspace_type = tensor_space_impl(impl)
    if dtype is None:
        dtype = default_dtype(impl)

    weighting = kwargs.pop('weighting', None)
    exponent = kwargs.pop('exponent', 2.0)
    if weighting is None and is_numeric_dtype(dtype):
        if exponent == float('inf') or partition.ndim == 0:
            weighting = 1.0
        else:
            weighting = partition.cell_volume

    tspace = tspace_type(partition.shape, dtype, exponent=exponent,
                         weighting=weighting)
    return DiscretizedSpace(partition, tspace, **kwargs)


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
    discr : `DiscretizedSpace`
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
    """
    if dtype is None:
        dtype = default_dtype(impl)

    nodes_on_bdry = kwargs.pop('nodes_on_bdry', False)
    partition = uniform_partition_fromintv(intv_prod, shape, nodes_on_bdry)
    return uniform_discr_frompartition(partition, dtype, impl, **kwargs)


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

        - ``None``: Use the cell volume as weighting constant (default).
        - ``float``: Weighting by a constant.
        - array-like: Point-wise weighting by an array.
        - `Weighting`: Use weighting class as-is. Compatibility
          with this space's elements is not checked during init.

    Returns
    -------
    discr : `DiscretizedSpace`
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
    discr : `DiscretizedSpace`
        Uniformly discretized space used as a template.
    min_pt, max_pt: float or sequence of floats, optional
        Desired minimum/maximum corners of the new space domain.
    shape : int or sequence of ints, optional
        Desired number of samples per axis of the new space.
    cell_sides : float or sequence of floats, optional
        Desired cell side lenghts of the new space's partition.

    Other Parameters
    ----------------
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
        Additional keyword parameters passed to the `DiscretizedSpace`
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
    if not isinstance(discr, DiscretizedSpace):
        raise TypeError('`discr` {!r} is not a DiscretizedSpace instance'
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

    return uniform_discr_frompartition(
        new_part, exponent=discr.exponent, impl=discr.impl, **kwargs
    )


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
