﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Lebesgue L^p type discretizations of function spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from builtins import str

import numpy as np
from numbers import Integral

from odl.discr.discretization import (
    DiscretizedSpace, DiscretizedSpaceElement, dspace_type)
from odl.discr.discr_mappings import (
    PointCollocation, NearestInterpolation, LinearInterpolation,
    PerAxisInterpolation)
from odl.discr.partition import (
    RectPartition, uniform_partition_fromintv, uniform_partition)
from odl.set import RealNumbers, ComplexNumbers, IntervalProd
from odl.space import FunctionSpace, ProductSpace
from odl.space.entry_points import tensor_space_impl
from odl.space.weighting import Weighting, NoWeighting, ConstWeighting
from odl.util import (
    apply_on_boundary, is_real_dtype, is_complex_floating_dtype,
    dtype_str, signature_string, indent_rows, is_string,
    normalized_scalar_param_list, safe_int_conv, normalized_nodes_on_bdry,
    broadcast_to)
from odl.util.ufuncs import DiscreteLpUfuncs

__all__ = ('DiscreteLp', 'DiscreteLpElement',
           'uniform_discr_frompartition', 'uniform_discr_fromspace',
           'uniform_discr_fromintv', 'uniform_discr',
           'uniform_discr_fromdiscr', 'discr_sequence_space')

_SUPPORTED_INTERP = ('nearest', 'linear')


class DiscreteLp(DiscretizedSpace):

    """Discretization of a Lebesgue :math:`L^p` space."""

    def __init__(self, fspace, partition, dspace, interp='nearest', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        fspace : `FunctionSpace`
            The continuous space to be discretized
        partition : `RectPartition`
            Partition of (a subset of) ``fspace.domain`` based on a
            `RectGrid`.
        dspace : `TensorSpace`
            Space of elements used for data storage. It must have the
            same `TensorSpace.field` as ``fspace``.
        interp : string or sequence of strings, optional
            The interpolation type to be used for discretization.
            A sequence is interpreted as interpolation scheme per
            axis.

            'nearest' : use nearest-neighbor interpolation (default)

            'linear' : use linear interpolation

        axis_labels : sequence of str, optional
            Names of the axes to use for plotting etc. Default:

                1D: ``['$x$']``

                2D: ``['$x$', '$y$']``

                3D: ``['$x$', '$y$', '$z$']``

                nD: ``['$x_1$', '$x_2$', ..., '$x_n$']``

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
        if fspace.out_dtype != dspace.dtype:
            raise ValueError('`out_dtype` of Function Space {} does not match '
                             'dspace dtype {}'
                             ''.format(fspace.out_dtype, dspace.dtype))

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
                raise ValueError('expected {} (ndim) entries in `interp`, '
                                 'got {}'.format(partition.ndim, len(interp)))

            self.__interp_byaxis = tuple(str(s).lower() for s in interp)
            if any(s not in _SUPPORTED_INTERP for s in self.interp_byaxis):
                raise ValueError('`interp` sequence {} contains illegal '
                                 'values'.format(interp))

        self.__partition = partition

        # Assign sampling and interpolation operators
        sampling = PointCollocation(fspace, self.partition, dspace)
        if all(s == 'nearest' for s in self.interp_byaxis):
            interpol = NearestInterpolation(fspace, self.partition, dspace)
        elif all(s == 'linear' for s in self.interp_byaxis):
            interpol = LinearInterpolation(fspace, self.partition, dspace)
        else:
            interpol = PerAxisInterpolation(
                fspace, self.partition, dspace, self.interp_by_axis)

        super(DiscreteLp, self).__init__(fspace, dspace, sampling, interpol)

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
            raise ValueError('unknown arguments {}'.format(kwargs))

    @property
    def interp(self):
        """Interpolation type of this discretization."""
        if all(interp == self.interp_byaxis[0]
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
        """All sampling points in the partition."""
        return self.partition.points()

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
        """If the weighting of the space is the same for all points."""
        try:
            is_uniformly_weighted = self.__is_uniformly_weighted
        except AttributeError:
            bdry_fracs = self.partition.boundary_cell_fractions
            is_uniformly_weighted = (
                np.allclose(bdry_fracs, 1.0) or
                self.exponent == float('inf') or
                not getattr(self.dspace, 'is_weighted', False))

            self.__is_uniformly_weighted = is_uniformly_weighted

        return is_uniformly_weighted

    def element(self, inp=None, **kwargs):
        """Create an element from ``inp`` or from scratch.

        Parameters
        ----------
        inp : optional
            Input data to create an element from.

            If ``inp`` is callable, it needs to be understood by the
            ``uspace.element`` method.

            Otherwise, it has to be understood by the ``dspace.element``
            method.
        kwargs :
            Additional arguments passed on to `sampling` when called
            on ``inp``, in the form ``sampling(inp, **kwargs)``.
            This can be used e.g. for functions with parameters.

        Other Parameters
        ----------------
        vectorized : bool
            Can only be used if ``inp`` is callable, in which case it
            indicates whether ``inp`` is vectorized. If not, it will be
            wrapped with a vectorizer.
            Default: True

        Returns
        -------
        element : `DiscreteLpElement`
            The discretized element, calculated as ``sampling(inp)`` or
            ``dspace.element(inp)``, tried in this order.

        Examples
        --------
        Elements can be created from array-like objects that represent
        an already discretized function:

        >>> space = odl.uniform_discr(-1, 1, 4)
        >>> space.element([1, 2, 3, 4])
        uniform_discr(-1.0, 1.0, 4).element([1.0, 2.0, 3.0, 4.0])
        >>> vector = odl.rn(4).element([0, 1, 2, 3])
        >>> space.element(vector)
        uniform_discr(-1.0, 1.0, 4).element([0.0, 1.0, 2.0, 3.0])

        On the other hand, non-discretized objects like Python functions
        can be discretized "on the fly":

        >>> space.element(lambda x: x * 2)
        uniform_discr(-1.0, 1.0, 4).element([-1.5, -0.5, 0.5, 1.5])

        This works also with parameterized functions, however only
        through keyword arguments (not positional arguments with
        defaults):

        >>> def f(x, **kwargs):
        ...     c = kwargs.pop('c', 0.0)
        ...     return np.maximum(x, c)
        ...
        >>> space = odl.uniform_discr(-1, 1, 4)
        >>> space.element(f, c=0.5)
        uniform_discr(-1.0, 1.0, 4).element([0.5, 0.5, 0.5, 0.75])

        See Also
        --------
        sampling : create a discrete element from an undiscretized one
        """
        if inp is None:
            return self.element_type(self, self.dspace.element())
        elif inp in self:
            return inp
        elif inp in self.dspace:
            return self.element_type(self, inp)
        elif callable(inp):
            vectorized = kwargs.pop('vectorized', True)
            # uspace element -> discretize
            inp_elem = self.uspace.element(inp, vectorized=vectorized)
            return self.element_type(self, self.sampling(inp_elem, **kwargs))
        else:
            # Sequence-type input
            arr = np.asarray(inp, dtype=self.dtype)
            input_shape = arr.shape

            # Attempt to handle some cases with exact match
            if arr.ndim > 1 and self.ndim == 1:
                arr = arr.squeeze()  # Squeeze could solve the problem
                if arr.shape != self.shape:
                    raise ValueError(
                        'input shape does not match grid shape: {} != {}'
                        ''.format(input_shape, self.shape))
            elif arr.shape != self.shape:
                # Try reshaping first, then broadcast
                try:
                    arr = arr.reshape(self.shape, order=self.order)
                except ValueError:
                    pass
                arr = broadcast_to(arr, self.shape)
            return self.element_type(self, self.dspace.element(arr))

    def _astype(self, dtype):
        """Internal helper for ``astype``."""
        fspace = self.uspace.astype(dtype)
        dspace = self.dspace.astype(dtype)
        return type(self)(fspace, self.partition, dspace, interp=self.interp,
                          axis_labels=self.axis_labels)

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

    def __repr__(self):
        """Return ``repr(self)``."""
        # Clunky check if the factory repr can be used
        if (uniform_partition_fromintv(
                self.uspace.domain, self.shape,
                nodes_on_bdry=False) == self.partition):
            use_uniform = True
            nodes_on_bdry = False
        elif (uniform_partition_fromintv(
                self.uspace.domain, self.shape,
                nodes_on_bdry=True) == self.partition):
            use_uniform = True
            nodes_on_bdry = True
        else:
            use_uniform = False

        if use_uniform:
            constructor = 'uniform_discr'
            if self.ndim == 1:
                posargs = [self.min_pt[0], self.max_pt[0], self.shape[0]]
            else:
                posargs = [list(a) for a in [self.min_pt, self.max_pt]]
                posargs.append(self.shape)

            default_dtype_s = dtype_str(
                self.dspace.default_dtype(RealNumbers()))

            if isinstance(self.weighting, NoWeighting):
                weighting = 'none'
            elif (isinstance(self.weighting, ConstWeighting) and
                  (np.isclose(self.weighting.const, self.cell_volume))):
                weighting = 'const'
            elif (self.ndim == 0 and
                  isinstance(self.weighting, ConstWeighting) and
                  (np.isclose(self.weighting.const, 1.0))):
                weighting = 'const'
            else:
                weighting = self.weighting

            dtype_s = dtype_str(self.dtype)
            optargs = [('interp', self.interp, 'nearest'),
                       ('impl', self.impl, 'numpy'),
                       ('nodes_on_bdry', nodes_on_bdry, False),
                       ('dtype', dtype_s, default_dtype_s),
                       ('order', self.order, 'C'),
                       ('weighting', weighting, 'const')]

            inner_str = signature_string(posargs, optargs,
                                         mod=[['!r'] * len(posargs),
                                              [''] * len(optargs)])
            return '{}({})'.format(constructor, inner_str)

        else:
            constructor = self.__class__.__name__
            posargs = [self.uspace, self.partition, self.dspace]

            optargs = [('interp', self.interp, 'nearest')]

            inner_str = signature_string(posargs, optargs,
                                         sep=[',\n', ', ', ',\n'],
                                         mod=['!r', '!s'])

            return '{}(\n{}\n)'.format(constructor, indent_rows(inner_str))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    @property
    def element_type(self):
        """`DiscreteLpElement`"""
        return DiscreteLpElement


class DiscreteLpElement(DiscretizedSpaceElement):

    """Representation of a `DiscreteLp` element."""

    def asarray(self, out=None):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype and
            shape.
        """
        if out is None:
            if self.space.ndim == 0:
                out = DiscretizedSpaceElement.asarray(self)
            else:
                out = DiscretizedSpaceElement.asarray(self).reshape(
                    self.shape, order=self.space.order)
            return out

        else:
            if out.shape not in (self.space.shape, (self.space.size,)):
                raise ValueError('output array has shape {}, expected '
                                 '{} or ({},)'
                                 ''.format(out.shape, self.space.shape,
                                           self.space.size))
            out_r = out.reshape(self.space.shape,
                                order=self.space.order)
            if out_r.flags.c_contiguous:
                out_order = 'C'
            elif out_r.flags.f_contiguous:
                out_order = 'F'
            else:
                raise ValueError('output array not contiguous')

            if out_order != self.space.order:
                raise ValueError('output array has ordering {!r}, '
                                 'expected {!r}'
                                 ''.format(self.space.order, out_order))

            super(DiscreteLpElement, self).asarray(
                out=out.ravel(order=self.space.order))
            return out

    @property
    def cell_sides(self):
        """Side lengths of a cell in an underlying *uniform* partition."""
        return self.space.cell_sides

    @property
    def cell_volume(self):
        """Cell volume of an underlying regular grid."""
        return self.space.cell_volume

    @property
    def real(self):
        """Real part of this element."""
        return self.space.real_space.element(self.ntuple.real)

    @real.setter
    def real(self, newreal):
        """Set the real part of this element to ``newreal``."""
        newreal_flat = np.asarray(newreal, order=self.space.order).reshape(
            -1, order=self.space.order)
        self.tensor.real = newreal_flat

    @property
    def imag(self):
        """Imaginary part of this element."""
        return self.space.real_space.element(self.ntuple.imag)

    @imag.setter
    def imag(self, newimag):
        """Set the imaginary part of this element to ``newimag``."""
        newimag_flat = np.asarray(newimag, order=self.space.order).reshape(
            -1, order=self.space.order)
        self.tensor.imag = newimag_flat

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
        >>> discr = uniform_discr(0, 1, 4, dtype='complex')
        >>> x = discr.element([5+1j, 3, 2-2j, 1j])
        >>> y = x.conj(); print(y)
        [(5-1j), (3-0j), (2+2j), -1j]

        The out parameter allows you to avoid a copy:

        >>> z = discr.element()
        >>> z_out = x.conj(out=z); print(z)
        [(5-1j), (3-0j), (2+2j), -1j]
        >>> z_out is z
        True

        It can also be used for in-place conjugation:

        >>> x_out = x.conj(out=x); print(x)
        [(5-1j), (3-0j), (2+2j), -1j]
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
            # For DiscretizedSetElement of the same type, use tensor directly
            self.tensor[indices] = values.tensor
        else:
            # Other sequence types are piped through a Numpy array. Equivalent
            # views are optimized for in Numpy.
            if indices == slice(None):
                values = np.atleast_1d(values)
                if (values.ndim > 1 and
                        values.shape != self.space.shape):
                    raise ValueError('shape {} of value array {} not equal '
                                     'to sampling grid shape {}'
                                     ''.format(values.shape, values,
                                               self.space.shape))
                values = values.ravel(order=self.space.order)

            super(DiscreteLpElement, self).__setitem__(indices, values)

    @property
    def ufuncs(self):
        """`DiscreteLpUfuncs`, access to numpy style ufuncs.

        Examples
        --------
        >>> X = uniform_discr(0, 1, 2)
        >>> x = X.element([1, -2])
        >>> x.ufuncs.absolute()
        uniform_discr(0.0, 1.0, 2).element([1.0, 2.0])

        These functions can also be used with broadcasting

        >>> x.ufuncs.add(3)
        uniform_discr(0.0, 1.0, 2).element([4.0, 1.0])

        and non-space elements

        >>> x.ufuncs.subtract([3, 3])
        uniform_discr(0.0, 1.0, 2).element([-2.0, -5.0])

        There is also support for various reductions (sum, prod, min, max)

        >>> x.ufuncs.sum()
        -1.0

        Also supports out parameter

        >>> y = X.element([3, 4])
        >>> out = X.element()
        >>> result = x.ufuncs.add(y, out=out)
        >>> result
        uniform_discr(0.0, 1.0, 2).element([4.0, 2.0])
        >>> result is out
        True

        Notes
        -----
        These are optimized to use the underlying tensor space and incur no
        overhead unless these do.
        """
        return DiscreteLpUfuncs(self)

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


def uniform_discr_frompartition(partition, exponent=2.0, interp='nearest',
                                impl='numpy', **kwargs):
    """Discretize an Lp function space given a uniform partition.

    Parameters
    ----------
    partition : `RectPartition`
        Regular (uniform) partition to be used for discretization.
    exponent : positive float, optional
        The parameter ``p`` in ``L^p``. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : string or sequence of strings, optional
        Interpolation type to be used for discretization.
        A sequence is interpreted as interpolation scheme per axis.
        Possible values:

            ``'nearest'`` : use nearest-neighbor interpolation.

            ``'linear'`` : use linear interpolation.

    impl : string, optional
        Implementation of the data storage arrays.

    Other Parameters
    ----------------
    order : {'C', 'F'}, optional
        Axis ordering in the data storage. Default: 'C'
    dtype : dtype
        Data type for the discretized space.

            Default for ``'numpy'``: ``'float64'`` / ``'complex128'``

            Default for ``'cuda'``: ``'float32'``

    weighting : {'const', 'none', `Weighting`}, optional
        Weighting of the discretized space functions.

            ``'const'`` : weight is a constant, the cell volume (default)

            ``'none'`` : no weighting

            `Weighting` : Use existing weighting as-is (not checked)

    kwargs :
        Additional keyword parameters passed to the `DiscreteLp`
        initializer.

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

    order = kwargs.pop('order', 'C')

    dtype = kwargs.pop('dtype', None)
    if dtype is not None:
        dtype = np.dtype(dtype)

    fspace = FunctionSpace(partition.set, out_dtype=dtype)
    ds_type = dspace_type(fspace, impl, dtype)

    if dtype is None:
        dtype = ds_type.default_dtype()

    weighting = kwargs.pop('weighting', 'const')
    if not isinstance(weighting, Weighting):
        weighting, weighting_in = str(weighting).lower(), weighting
        if weighting == 'none' or float(exponent) == float('inf'):
            weighting = None
        elif weighting == 'const':
            weighting = 1.0 if partition.ndim == 0 else partition.cell_volume
        else:
            raise ValueError("`weighting` '{}' not understood"
                             "".format(weighting_in))

    dspace = ds_type(partition.shape, dtype=dtype, impl=impl, order=order,
                     exponent=exponent, weighting=weighting)
    return DiscreteLp(fspace, partition, dspace, interp, **kwargs)


def uniform_discr_fromspace(fspace, shape, exponent=2.0, interp='nearest',
                            impl='numpy', **kwargs):
    """Discretize an Lp function space by uniform partition.

    Parameters
    ----------
    fspace : `FunctionSpace`
        Continuous function space. Its domain must be an
        `IntervalProd` instance.
    shape : int or sequence of ints
        Number of samples per axis.
    exponent : positive float, optional
        The parameter ``p`` in ``L^p``. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : string or sequence of strings, optional
        Interpolation type to be used for discretization.
        A sequence is interpreted as interpolation scheme per axis.
        Possible values:

            ``'nearest'`` : use nearest-neighbor interpolation.

            ``'linear'`` : use linear interpolation.

    impl : string, optional
        Implementation of the data storage arrays

    Other Parameters
    ----------------
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

    order : {'C', 'F'}, optional
        Axis ordering in the data storage. Default: 'C'
    dtype : dtype, optional
        Data type for the discretized space. If not specified, the
        `FunctionSpace.out_dtype` of ``fspace`` is used.
    weighting : {'const', 'none', `Weighting`}, optional
        Weighting of the discretized space functions.

            ``'const'`` : weight is a constant, the cell volume (default)

            ``'none'`` : no weighting

            `Weighting` : Use existing weighting as-is (not checked)

    kwargs :
        Additional keyword parameters passed to the `DiscreteLp`
        initializer.

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    >>> intv = odl.IntervalProd(0, 1)
    >>> space = odl.FunctionSpace(intv)
    >>> uniform_discr_fromspace(space, 10)
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

    dtype = kwargs.pop('dtype', None)

    # Set data type. If given check consistency with fspace's field and
    # out_dtype. If not given, take the latter.
    if dtype is None:
        dtype = fspace.out_dtype
    else:
        dtype, dtype_in = np.dtype(dtype), dtype
        if not np.can_cast(fspace.out_dtype, dtype, casting='safe'):
            raise ValueError('cannot safely cast from output data {} type of '
                             'the function space to given data type {}'
                             ''.format(fspace.out, dtype_in))

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
    partition = uniform_partition_fromintv(fspace.domain, shape,
                                           nodes_on_bdry)

    return uniform_discr_frompartition(
        partition=partition, exponent=exponent, interp=interp, impl=impl,
        dtype=dtype, **kwargs)


def uniform_discr_fromintv(interval, shape, exponent=2.0, interp='nearest',
                           impl='numpy', **kwargs):
    """Discretize an Lp function space by uniform sampling.

    Parameters
    ----------
    interval : `IntervalProd`
        The domain of the uniformly discretized space.
    shape : int or sequence of ints
        Number of samples per axis.
    exponent : positive float, optional
        The parameter ``p`` in ``L^p``. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : string or sequence of strings, optional
        Interpolation type to be used for discretization.
        A sequence is interpreted as interpolation scheme per axis.
        Possible values:

            ``'nearest'`` : use nearest-neighbor interpolation.

            ``'linear'`` : use linear interpolation.

    impl : str, optional
        Implementation of the data storage arrays.

    Other Parameters
    ----------------
    nodes_on_bdry : bool or sequence, optional
        If a sequence is provided, it determines per axis whether to
        place the last grid point on the boundary (``True``) or shift it
        by half a cell size into the interior (``False``). In each axis,
        an entry may consist in a single bool or a 2-tuple of
        bool. In the latter case, the first tuple entry decides for
        the left, the second for the right boundary. The length of the
        sequence must be ``array.ndim``.

        A single boolean is interpreted as a global choice for all
        boundaries.
        Default: ``False``

    order : {'C', 'F'}, optional
        Axis ordering in the data storage. Default: 'C'
    dtype : dtype
        Data type for the discretized space.

            Default for ``'numpy'``: ``'float64'`` / ``'complex128'``

            Default for ``'cuda'``: ``'float32'``

    weighting : {'const', 'none', `Weighting`}, optional
        Weighting of the discretized space functions.

            ``'const'`` : weight is a constant, the cell volume (default)

            ``'none'`` : no weighting

            `Weighting` : Use existing weighting as-is (not checked)

    kwargs :
        Additional keyword parameters passed to the `DiscreteLp`
        initializer.

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
    dtype = kwargs.pop('dtype', None)
    if dtype is None:
        dtype = tensor_space_impl(str(impl).lower()).default_dtype()

    fspace = FunctionSpace(interval, out_dtype=dtype)
    return uniform_discr_fromspace(fspace, shape, exponent, interp, impl,
                                   **kwargs)


def uniform_discr(min_pt, max_pt, shape, exponent=2.0, interp='nearest',
                  impl='numpy', **kwargs):
    """Discretize an Lp function space by uniform sampling.

    Parameters
    ----------
    min_pt : float or sequence of floats
        Maximum corners of the desired function domain.
    max_pt : float or sequence of floats
        Minimum corners of the desired function domain.
    shape : int or sequence of ints
        Number of samples per axis.
    exponent : positive float, optional
        The parameter :math:`p` in :math:`L^p`. If the exponent is not
        equal to the default 2.0, the space has no inner product.
    interp : string or sequence of strings, optional
        Interpolation type to be used for discretization.
        A sequence is interpreted as interpolation scheme per axis.
        Possible values:

            ``'nearest'`` : use nearest-neighbor interpolation.

            ``'linear'`` : use linear interpolation.

    impl : string, optional
        Implementation of the data storage arrays.

    Other Parameters
    ----------------
    dtype : dtype
        Data type for the discretized space.

            Default for ``'numpy'``: ``'float64'`` / ``'complex128'``

            Default for ``'cuda'``: ``'float32'``

    order : {'C', 'F'}, optional
        Ordering of the axes in the data storage. ``'C'`` means the
        first axis varies slowest, the last axis fastest;
        vice versa for ``'F'``.
        Default: ``'C'``
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

    weighting : {'const', 'none'}, optional
        Weighting of the discretized space functions.

            'const' : weight is a constant, the cell volume (default)

            'none' : no weighting

    Returns
    -------
    discr : `DiscreteLp`
        The uniformly discretized function space

    Examples
    --------
    Create real space:

    >>> space = uniform_discr([0, 0], [1, 1], (10, 10))
    >>> space
    uniform_discr([0.0, 0.0], [1.0, 1.0], (10, 10))
    >>> space.cell_sides
    array([ 0.1,  0.1])
    >>> space.dtype
    dtype('float64')
    >>> space.is_real_space
    True

    Create complex space by giving a dtype:

    >>> space = uniform_discr([0, 0], [1, 1], (10, 10), dtype='complex')
    >>> space
    uniform_discr([0.0, 0.0], [1.0, 1.0], (10, 10), dtype='complex')
    >>> space.is_complex_space
    True
    >>> space.real_space  # Get real counterpart
    uniform_discr([0.0, 0.0], [1.0, 1.0], (10, 10))

    See Also
    --------
    uniform_discr_frompartition : uniform Lp discretization using a given
        uniform partition of a function domain
    uniform_discr_fromspace : uniform discretization from an existing
        function space
    uniform_discr_fromintv : uniform discretization from an existing
        interval product
    """
    interval = IntervalProd(min_pt, max_pt)
    return uniform_discr_fromintv(interval, shape, exponent, interp, impl,
                                  **kwargs)


def discr_sequence_space(shape, exponent=2.0, impl='numpy', **kwargs):
    """Return an object mimicing the sequence space ``l^p(R^d)``.

    The returned object is a `DiscreteLp` without sampling and
    interpolation operators. It uses a grid with stride 1 and no
    weighting.

    Parameters
    ----------
    shape : int or sequence of ints
        Number of element entries per axis.
    exponent : positive float, optional
        The parameter ``p`` in ```l^p``. If the exponent is
        not equal to the default 2.0, the space has no inner
        product.
    impl : string, optional
        Implementation of the data storage arrays.
    dtype : dtype
        Data type for the discretized space.

            Default for ``'numpy'``: ``'float64'`` / ``'complex128'``

            Default for ``'cuda'``: ``'float32'``

    order : {'C', 'F'}, optional
        Axis ordering in the data storage. Default: 'C'

    kwargs :
        Additional keyword parameters passed to the `DiscreteLp`
        initializer.

    Returns
    -------
    seqspc : `DiscreteLp`
        Sequence-space-like discrete Lp.

    Examples
    --------
    >>> seq_spc = discr_sequence_space((3, 3))
    >>> seq_spc.one().norm() == 3.0
    True
    >>> seq_spc = discr_sequence_space((3, 3), exponent=1)
    >>> seq_spc.one().norm() == 9.0
    True
    """
    kwargs.pop('weighting', None)
    kwargs.pop('nodes_on_bdry', None)
    shape = np.atleast_1d(shape)
    return uniform_discr([0] * len(shape), shape - 1, shape, impl=impl,
                         exponent=exponent, nodes_on_bdry=True,
                         weighting='none', **kwargs)


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
    uniform_discr([1.0, 1.0], [2.0, 3.0], (10, 5))
    >>> odl.uniform_discr_fromdiscr(discr, max_pt=[0, 0])
    uniform_discr([-1.0, -2.0], [0.0, 0.0], (10, 5))
    >>> odl.uniform_discr_fromdiscr(discr, cell_sides=[1, 1])
    uniform_discr([0.0, 0.0], [1.0, 2.0], (1, 2))
    >>> odl.uniform_discr_fromdiscr(discr, shape=[5, 5])
    uniform_discr([0.0, 0.0], [1.0, 2.0], (5, 5))
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
    uniform_discr([2.0, 1.0], [3.0, 2.25], (10, 5))
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
    uniform_discr([2.5, 1.0], [3.0, 2.25], (5, 5))
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
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
