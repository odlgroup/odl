# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Base classes for implementations of tensor spaces."""

from __future__ import absolute_import, division, print_function

from numbers import Integral

import numpy as np

from odl.set.sets import ComplexNumbers, RealNumbers
from odl.set.space import LinearSpace
from odl.util import (
    dtype_str, is_complex_floating_dtype, is_floating_dtype,
    is_numeric_dtype, is_real_dtype, is_real_floating_dtype, safe_int_conv,
    signature_string)
from odl.util.utility import TYPE_MAP_C2R, TYPE_MAP_R2C

__all__ = ('TensorSpace',)


class TensorSpace(LinearSpace):

    """Base class for sets of tensors of arbitrary data type.

    A tensor is, in the most general sense, a multi-dimensional array
    that allows operations per entry (keep the rank constant),
    reductions / contractions (reduce the rank) and broadcasting
    (raises the rank).
    For non-numeric data type like ``object``, the range of valid
    operations is rather limited since such a set of tensors does not
    define a vector space.
    Any numeric data type, on the other hand, is considered valid for
    a tensor space, although certain operations - like division with
    integer dtype - are not guaranteed to yield reasonable results.

    Under these restrictions, all basic vector space operations are
    supported by this class, along with reductions based on arithmetic
    or comparison, and element-wise mathematical functions ("ufuncs").

    See the `Wikipedia article on tensors`_ for further details.
    See also [Hac2012] "Part I Algebraic Tensors" for a rigorous
    treatment of tensors with a definition close to this one.

    Note also that this notion of tensors is the same as in popular
    Deep Learning frameworks.

    References
    ----------
    [Hac2012] Hackbusch, W. *Tensor Spaces and Numerical Tensor Calculus*.
    Springer, 2012.

    .. _Wikipedia article on tensors: https://en.wikipedia.org/wiki/Tensor
    """

    def __init__(self, shape, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        shape : nonnegative int or sequence of nonnegative ints
            Number of entries of type ``dtype`` per axis in this space. A
            single integer results in a space with rank 1, i.e., 1 axis.
        dtype :
            Data type of elements in this space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string.
            For a data type with a ``dtype.shape``, these extra dimensions
            are added *to the left* of ``shape``.
        """
        # Handle shape and dtype, taking care also of dtypes with shape
        try:
            shape, shape_in = tuple(safe_int_conv(s) for s in shape), shape
        except TypeError:
            shape, shape_in = (safe_int_conv(shape),), shape
        if any(s < 0 for s in shape):
            raise ValueError('`shape` must have only nonnegative entries, got '
                             '{}'.format(shape_in))
        dtype = np.dtype(dtype)

        # We choose this order in contrast to Numpy, since we usually want
        # to represent discretizations of vector- or tensor-valued functions,
        # i.e., if dtype.shape == (3,) we expect f[0] to have shape `shape`.
        self.__shape = dtype.shape + shape
        self.__dtype = dtype.base

        if is_real_dtype(self.dtype):
            # real includes non-floating-point like integers
            field = RealNumbers()
            self.__real_dtype = self.dtype
            self.__real_space = self
            self.__complex_dtype = TYPE_MAP_R2C.get(self.dtype, None)
            self.__complex_space = None  # Set in first call of astype
        elif is_complex_floating_dtype(self.dtype):
            field = ComplexNumbers()
            self.__real_dtype = TYPE_MAP_C2R[self.dtype]
            self.__real_space = None  # Set in first call of astype
            self.__complex_dtype = self.dtype
            self.__complex_space = self
        else:
            field = None

        LinearSpace.__init__(self, field)

    @property
    def impl(self):
        """Name of the implementation back-end of this tensor set.

        This property should be overridden by subclasses.
        """
        raise NotImplementedError('abstract method')

    @property
    def shape(self):
        """Number of scalar elements per axis.

        .. note::
            If `dtype` has a shape, we add it to the **left** of the given
            ``shape`` in the class creation. This is in contrast to NumPy,
            which adds extra axes to the **right**. We do this since we
            usually want to represent discretizations of vector- or
            tensor-valued functions by this, i.e., if
            ``dtype.shape == (3,)`` we expect ``f[0]`` to have shape
            ``shape``.
        """
        return self.__shape

    @property
    def dtype(self):
        """Scalar data type of each entry in an element of this space."""
        return self.__dtype

    @property
    def is_real(self):
        """True if this is a space of real tensors."""
        return is_real_floating_dtype(self.dtype)

    @property
    def is_complex(self):
        """True if this is a space of complex tensors."""
        return is_complex_floating_dtype(self.dtype)

    @property
    def real_dtype(self):
        """The real dtype corresponding to this space's `dtype`.

        Raises
        ------
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise NotImplementedError(
                '`real_dtype` not defined for non-numeric `dtype`')
        return self.__real_dtype

    @property
    def complex_dtype(self):
        """The complex dtype corresponding to this space's `dtype`.

        Raises
        ------
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise NotImplementedError(
                '`complex_dtype` not defined for non-numeric `dtype`')
        return self.__complex_dtype

    @property
    def real_space(self):
        """The space corresponding to this space's `real_dtype`.

        Raises
        ------
        ValueError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise ValueError(
                '`real_space` not defined for non-numeric `dtype`')
        return self.astype(self.real_dtype)

    @property
    def complex_space(self):
        """The space corresponding to this space's `complex_dtype`.

        Raises
        ------
        ValueError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
            raise ValueError(
                '`complex_space` not defined for non-numeric `dtype`')
        return self.astype(self.complex_dtype)

    def _astype(self, dtype):
        """Internal helper for `astype`.

        Subclasses with differing init parameters should overload this
        method.
        """
        kwargs = {}
        if is_floating_dtype(dtype):
            # Use weighting only for floating-point types, otherwise, e.g.,
            # `space.astype(bool)` would fail
            weighting = getattr(self, 'weighting', None)
            if weighting is not None:
                kwargs['weighting'] = weighting

        return type(self)(self.shape, dtype=dtype, **kwargs)

    def astype(self, dtype):
        """Return a copy of this space with new ``dtype``.

        Parameters
        ----------
        dtype :
            Scalar data type of the returned space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.

        Returns
        -------
        newspace : `TensorSpace`
            Version of this space with given data type.
        """
        if dtype is None:
            # Need to filter this out since Numpy iterprets it as 'float'
            raise ValueError('`None` is not a valid data type')

        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self

        if is_numeric_dtype(self.dtype):
            # Caching for real and complex versions (exact dtype mappings)
            if dtype == self.__real_dtype:
                if self.__real_space is None:
                    self.__real_space = self._astype(dtype)
                return self.__real_space
            elif dtype == self.__complex_dtype:
                if self.__complex_space is None:
                    self.__complex_space = self._astype(dtype)
                return self.__complex_space
            else:
                return self._astype(dtype)
        else:
            return self._astype(dtype)

    @property
    def default_order(self):
        """Default storage order for new elements in this space.

        This property should be overridden by subclasses.
        """
        raise NotImplementedError('abstract method')

    @property
    def size(self):
        """Total number of entries in an element of this space."""
        return (0 if self.shape == () else
                int(np.prod(self.shape, dtype='int64')))

    @property
    def ndim(self):
        """Number of axes (=dimensions) of this space, also called "rank"."""
        return len(self.shape)

    def __len__(self):
        """Number of tensor entries along the first axis."""
        return int(self.shape[0])

    @property
    def itemsize(self):
        """Size in bytes of one entry in an element of this space."""
        return int(self.dtype.itemsize)

    @property
    def nbytes(self):
        """Total number of bytes in memory used by an element of this space."""
        return self.size * self.itemsize

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if ``self`` and ``other`` have the same type, `shape`
            and `dtype`, otherwise ``False``.

        Examples
        --------
        Sizes, data types and other essential properties characterize
        spaces and decide about equality:

        >>> spc = odl.tensor_space(3, dtype='uint64')
        >>> spc == spc
        True
        >>> spc2 = odl.tensor_space(3, dtype='uint64')
        >>> spc2 == spc
        True
        >>> smaller_spc = odl.tensor_space(2, dtype='uint64')
        >>> spc == smaller_spc
        False
        >>> other_dtype_spc = odl.tensor_space(3, dtype='uint32')
        >>> spc == other_dtype_spc
        False
        >>> other_shape_spc = odl.tensor_space((3, 1), dtype='uint64')
        >>> spc == other_shape_spc
        False
        """
        if other is self:
            return True

        return (type(other) is type(self) and
                self.shape == other.shape and
                self.dtype == other.dtype)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.shape, self.dtype))

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.shape, dtype_str(self.dtype)]
        return "{}({})".format(self.__class__.__name__,
                               signature_string(posargs, []))

    def __str__(self):
        """Return ``str(self)``."""
        return repr(self)

    @property
    def examples(self):
        """Return example random vectors."""
        # Always return the same numbers
        rand_state = np.random.get_state()
        np.random.seed(1337)

        if is_numeric_dtype(self.dtype):
            yield ('Linearly spaced samples', self.element(
                np.linspace(0, 1, self.size).reshape(self.shape)))
            yield ('Normally distributed noise',
                   self.element(np.random.standard_normal(self.shape)))

        if self.is_real:
            yield ('Uniformly distributed noise',
                   self.element(np.random.uniform(size=self.shape)))
        elif self.is_complex:
            yield ('Uniformly distributed noise',
                   self.element(np.random.uniform(size=self.shape) +
                                np.random.uniform(size=self.shape) * 1j))
        else:
            # TODO: return something that always works, like zeros or ones?
            raise NotImplementedError('no examples available for non-numeric'
                                      'data type')

        np.random.set_state(rand_state)

    def zero(self):
        """Return a tensor of all zeros.

        This method should be overridden by subclasses.

        Returns
        -------
        zero : `Tensor`
            A tensor of all zeros.
        """
        raise NotImplementedError('abstract method')

    def one(self):
        """Return a tensor of all ones.

        This method should be overridden by subclasses.

        Returns
        -------
        one : `Tensor`
            A tensor of all one.
        """
        raise NotImplementedError('abstract method')

    def _multiply(self, x1, x2, out):
        """The entry-wise product of two tensors, assigned to ``out``.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError('abstract method')

    def _divide(self, x1, x2, out):
        """The entry-wise quotient of two tensors, assigned to ``out``.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError('abstract method')

    @staticmethod
    def default_dtype(field=None):
        """Return the default data type for a given field.

        This method should be overridden by subclasses.

        Parameters
        ----------
        field : `Field`, optional
            Set of numbers to be represented by a data type.

        Returns
        -------
        dtype :
            Numpy data type specifier.
        """
        raise NotImplementedError('abstract method')

    @staticmethod
    def available_dtypes():
        """Return the set of data types available in this implementation.

        This method should be overridden by subclasses.
        """
        raise NotImplementedError('abstract method')

    def show(self, elem, title=None, method='', indices=None, force_show=False,
             fig=None, **kwargs):
        """Display the function graphically.

        Parameters
        ----------
        elem : array-like
            Element to display using the properties of this space.
        title : string, optional
            Set the title of the figure
        method : string, optional
            1d methods:

            - ``'plot'`` : graph plot

            - ``'scatter'`` : scattered 2d points (2nd axis <-> value)

            2d methods:

            - ``'imshow'`` : image plot with coloring according to value,
              including a colorbar.

            - ``'scatter'`` : cloud of scattered 3d points (3rd axis <-> value)

        indices : index expression, optional
            Display a slice of the array instead of the full array. The
            index expression is most easily created with the `numpy.s_`
            constructor, i.e. supply ``np.s_[:, 1, :]`` to display the
            first slice along the second axis.
            For data with 3 or more dimensions, the 2d slice in the first
            two axes at the "middle" along the remaining axes is shown
            (semantically ``[:, :, shape[2:] // 2]``).
            This option is mutually exclusive to ``coords``.
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
        from odl.discr import uniform_grid
        from odl.util.graphics import show_discrete_data

        elem = self.element(elem)

        # Default to showing x-y slice "in the middle"
        if indices is None and self.ndim >= 3:
            indices = tuple(
                [slice(None)] * 2 + [n // 2 for n in self.shape[2:]]
            )

        if isinstance(indices, (Integral, slice)):
            indices = (indices,)
        elif indices is None or indices == Ellipsis:
            indices = (slice(None),) * self.ndim
        else:
            indices = tuple(indices)

        # Replace None by slice(None)
        indices = tuple(slice(None) if idx is None else idx for idx in indices)

        if Ellipsis in indices:
            # Replace Ellipsis with the correct number of [:] expressions
            pos = indices.index(Ellipsis)
            indices = (indices[:pos] +
                       (np.s_[:], ) * (self.ndim - len(indices) + 1) +
                       indices[pos + 1:])

        if len(indices) < self.ndim:
            raise ValueError(
                'too few axes ({} < {})'.format(len(indices), self.ndim)
            )
        if len(indices) > self.ndim:
            raise ValueError(
                'too many axes ({} > {})'.format(len(indices), self.ndim)
            )

        # Squeeze grid and values according to the index expression
        full_grid = uniform_grid(
            [0] * self.ndim, np.array(self.shape) - 1, self.shape
        )
        grid = full_grid[indices].squeeze()
        values = elem[indices].squeeze()

        return show_discrete_data(values, grid, title=title, method=method,
                                  force_show=force_show, fig=fig, **kwargs)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
