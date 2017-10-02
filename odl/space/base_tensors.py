﻿# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Base classes for implementations of tensor spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

import numpy as np

from odl.set.sets import Set, RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace, LinearSpaceElement
from odl.util import (
    is_scalar_dtype, is_floating_dtype, is_real_dtype,
    is_complex_floating_dtype, safe_int_conv,
    arraynd_repr, arraynd_str, dtype_str, signature_string, indent_rows)
from odl.util.ufuncs import TensorSpaceUfuncs
from odl.util.utility import TYPE_MAP_R2C, TYPE_MAP_C2R


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

    See `this Wikipedia article <https://en.wikipedia.org/wiki/Tensor>`_
    on tensors for further details.
    """

    def __init__(self, shape, dtype, order='K'):
        """Initialize a new instance.

        Parameters
        ----------
        shape : positive int or sequence of positive ints
            Number of entries per axis for elements in this space. A
            single integer results in a space with rank 1, i.e., 1 axis.
        dtype :
            Scalar data type of elements in this space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.
        order : {'K', 'C', 'F'}, optional
            Axis ordering of the data storage. Only relevant for more
            than 1 axis.
            For ``'C'`` and ``'F'``, elements are forced to use
            contiguous memory in the respective ordering.
            For ``'K'`` no contiguousness is enforced.
        """
        try:
            self.__shape = tuple(safe_int_conv(s) for s in shape)
        except TypeError:
            self.__shape = (safe_int_conv(shape),)
        if any(s < 0 for s in self.shape):
            raise ValueError('`shape` must have only positive entries, got '
                             '{}'.format(shape))

        self.__dtype = np.dtype(dtype)
        if self.dtype.shape:
            raise ValueError('`dtype` {!r} with shape {} not allowed'
                             ''.format(dtype, self.dtype.shape))
        self.__order = str(order).upper()
        if self.order not in ('K', 'C', 'F'):
            raise ValueError("`order '{}' not understood".format(order))

        if is_real_dtype(self.dtype):
            # real includes non-floating-point like integers
            field = RealNumbers()
            self.__is_real = True
            self.__is_complex = False
            self.__real_dtype = self.dtype
            self.__real_space = self
            self.__complex_dtype = TYPE_MAP_R2C.get(self.dtype, None)
            self.__complex_space = None  # Set in first call of astype
        elif is_complex_floating_dtype(self.dtype):
            field = ComplexNumbers()
            self.__is_real = False
            self.__is_complex = True
            self.__real_dtype = TYPE_MAP_C2R[self.dtype]
            self.__real_space = None  # Set in first call of astype
            self.__complex_dtype = self.dtype
            self.__complex_space = self
        else:
            field = None
            self.__is_real = False
            self.__is_complex = False

        LinearSpace.__init__(self, field)
        self.__is_floating = is_floating_dtype(self.dtype)

    @property
    def impl(self):
        """Name of the implementation back-end of this tensor set."""
        raise NotImplementedError('abstract method')

    @property
    def shape(self):
        """Number of elements per axis."""
        return self.__shape

    @property
    def dtype(self):
        """Scalar data type of each entry in an element of this space."""
        return self.__dtype

    @property
    def is_numeric(self):
        """``True`` if `dtype` is numeric, otherwise ``False``."""
        return self.__is_real or self.__is_complex

    @property
    def is_real_space(self):
        """True if this is a space of real tensors."""
        return self.__is_real and self.__is_floating

    @property
    def is_complex_space(self):
        """True if this is a space of complex tensors."""
        return self.__is_complex and self.__is_floating

    @property
    def real_dtype(self):
        """The real dtype corresponding to this space's `dtype`.

        Raises
        ------
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not self.is_numeric:
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
        if not self.is_numeric:
            raise NotImplementedError(
                '`complex_dtype` not defined for non-numeric `dtype`')
        return self.__complex_dtype

    @property
    def real_space(self):
        """The space corresponding to this space's `real_dtype`.

        Raises
        ------
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not self.is_numeric:
            raise NotImplementedError(
                '`real_space` not defined for non-numeric `dtype`')
        return self.astype(self.real_dtype)

    @property
    def complex_space(self):
        """The space corresponding to this space's `complex_dtype`.

        Raises
        ------
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not self.is_numeric:
            raise NotImplementedError(
                '`complex_space` not defined for non-numeric `dtype`')
        return self.astype(self.complex_dtype)

    def _astype(self, dtype, order):
        """Internal helper for `astype`.

        Subclasses with differing init parameters should overload this
        method.
        """
        return type(self)(self.shape, dtype=dtype, order=order,
                          weighting=getattr(self, 'weighting', None))

    def astype(self, dtype, order=None):
        """Return a copy of this space with new ``dtype``.

        Parameters
        ----------
        dtype :
            Scalar data type of the returned space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string. Data types with non-trivial
            shapes are not allowed.
        order : {'K', 'C', 'F'}, optional
            Axis ordering of the data storage. Only relevant for more
            than 1 axis.
            For ``'C'`` and ``'F'``, elements are forced to use
            contiguous memory in the respective ordering.
            For ``'K'`` no contiguousness is enforced.
            The default ``None`` is equivalent to ``self.order``.

        Returns
        -------
        newspace : `TensorSpace`
            Version of this space with given data type.
        """
        if dtype is None:
            # Need to filter this out since Numpy iterprets it as 'float'
            raise ValueError('`None` is not a valid data type')

        if order is None:
            order = self.order

        dtype = np.dtype(dtype)
        if dtype == self.dtype and order == self.order:
            return self

        if self.is_numeric:
            # Caching for real and complex versions (exact dtype mappings)
            if dtype == self.__real_dtype and order == self.order:
                if self.__real_space is None:
                    self.__real_space = self._astype(dtype, self.order)
                return self.__real_space
            elif dtype == self.__complex_dtype and order == self.order:
                if self.__complex_space is None:
                    self.__complex_space = self._astype(dtype, self.order)
                return self.__complex_space
            else:
                return self._astype(dtype, order)
        else:
            return self._astype(dtype, order)

    @property
    def order(self):
        """Guaranteed data storage order in this space.

        This is one of ``('C', 'F', 'K')``, where ``'K'`` means
        "no guarantee".
        """
        return self.__order

    @property
    def new_elem_order(self):
        """Storage order for new elements in this space.

        This is identical to `order` except for ``self.order == 'K'``,
        where ``'C'`` is returned.
        """
        return 'C' if self.order == 'K' else self.order

    @property
    def view_order(self):
        """Order argument for view-preserving operations.

        This is identical to `order` except for ``self.order == 'K'``,
        where ``'A'`` ("any") is returned. The main use case for this
        property is in reshaping with `numpy.reshape` or `numpy.ravel`,
        if views into the original array should be preserved.
        """
        return 'A' if self.order == 'K' else self.order

    @property
    def size(self):
        """Total number of entries in an element of this space."""
        return np.prod(self.shape)

    @property
    def ndim(self):
        """Number of axes (=dimensions) of this space, also called "rank"."""
        return len(self.shape)

    def __len__(self):
        """Number of tensor entries along the first axis."""
        return self.shape[0]

    @property
    def itemsize(self):
        """Size in bytes of one entry in an element of this space."""
        return self.dtype.itemsize

    @property
    def nbytes(self):
        """Total number of bytes used by an element of this space in memory."""
        return self.size * self.itemsize

    def __contains__(self, other):
        """Return ``other in self``.

        Returns
        -------
        contains : bool
            ``True`` if ``other`` has a ``space`` attribute that is equal
            to this space, ``False`` otherwise.

        Examples
        --------
        Elements created with the `TensorSpace.element` method are
        guaranteed to be contained in the same space:

        >>> spc = odl.tensor_space((2, 3), dtype='uint64')
        >>> spc.element() in spc
        True
        >>> x = spc.element([[0, 1, 2],
        ...                  [3, 4, 5]])
        >>> x in spc
        True

        Sizes, data types and other essential properties characterize
        spaces and decide about membership:

        >>> smaller_spc = odl.tensor_space((2, 2), dtype='uint64')
        >>> y = smaller_spc.element([[0, 1],
        ...                          [2, 3]])
        >>> y in spc
        False
        >>> x in smaller_spc
        False
        >>> other_dtype_spc = odl.tensor_space((2, 3), dtype='uint32')
        >>> z = other_dtype_spc.element([[0, 1, 2],
        ...                              [3, 4, 5]])
        >>> z in spc
        False
        >>> x in other_dtype_spc
        False

        On the other hand, spaces are not unique:

        >>> spc2 = odl.tensor_space((2, 3), dtype='uint64')  # same as spc
        >>> x2 = spc2.element([[5, 4, 3],
        ...                    [2, 1, 0]])
        >>> x2 in spc
        True
        >>> x in spc2
        True

        Of course, random garbage is not in the space:

        >>> spc = odl.tensor_space((2, 3), dtype='uint64')
        >>> None in spc
        False
        >>> object in spc
        False
        >>> False in spc
        False
        """
        return getattr(other, 'space', None) == self

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if ``self`` and ``other`` have the same type, `shape`
            `dtype` and `order`, otherwise ``False``.

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

        return (type(self) == type(other) and
                self.shape == other.shape and
                self.dtype == other.dtype and
                self.order == other.order)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.shape, self.dtype, self.order))

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.shape, dtype_str(self.dtype)]
        optargs = [('order', self.order, 'K')]
        return "{}({})".format(self.__class__.__name__,
                               signature_string(posargs, optargs))

    __str__ = __repr__

    @property
    def examples(self):
        """Return example random vectors."""
        # Always return the same numbers
        rand_state = np.random.get_state()
        np.random.seed(1337)

        if self.is_numeric:
            yield ('Linearly spaced samples', self.element(
                np.linspace(0, 1, self.size).reshape(self.shape)))
            yield ('Normally distributed noise',
                   self.element(np.random.standard_normal(self.shape)))

        if self.is_real_space:
            yield ('Uniformly distributed noise',
                   self.element(np.random.uniform(size=self.shape)))
        elif self.is_complex_space:
            yield ('Uniformly distributed noise',
                   self.element(np.random.uniform(size=self.shape) +
                                np.random.uniform(size=self.shape) * 1j))
        else:
            # TODO: return something that always works, like zeros or ones?
            raise NotImplementedError('no examples available for non-numeric'
                                      'data type')

        np.random.set_state(rand_state)

    def zero(self):
        """Return a tensor of all zeros."""
        raise NotImplementedError('abstract method')

    def one(self):
        """Return a tensor of all ones."""
        raise NotImplementedError('abstract method')

    def _multiply(self, x1, x2, out):
        """The entry-wise product of two tensors, assigned to ``out``."""
        raise NotImplementedError('abstract method')

    def _divide(self, x1, x2, out):
        """The entry-wise quotient of two tensors, assigned to ``out``."""
        raise NotImplementedError('abstract method')

    @staticmethod
    def default_dtype(field=None):
        """Return the default data type for a given field.

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
        """Return the list of data types available in this implementation."""
        raise NotImplementedError('abstract method')

    @property
    def element_type(self):
        """Type of elements in this set: `Tensor`."""
        return Tensor


class Tensor(LinearSpaceElement):

    """Abstract class for representation of `TensorSpace` elements."""

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : TensorSpace
            The space to which this tensor belongs.
        """
        assert isinstance(space, TensorSpace)
        self.__space = space

    @property
    def impl(self):
        """Name of the implementation back-end of this tensor."""
        return self.space.impl

    copy = LinearSpaceElement.copy

    def asarray(self, out=None):
        """Extract the data of this tensor as a Numpy array.

        Parameters
        ----------
        out : `numpy.ndarray`
            Array to write the result to.

        Returns
        -------
        asarray : `numpy.ndarray`
            Numpy array of the same data type and shape as the space.
            If ``out`` was given, the returned object is a reference
            to it.
        """
        raise NotImplementedError('abstract method')

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be accessed.

        Returns
        -------
        values : `TensorSpace.dtype` or `Tensor`
            The value(s) at the given indices. Note that depending on
            the implementation, the returned object may be a (writable)
            view into the original array.
        """
        raise NotImplementedError('abstract method')

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be written to.
        values : scalar, `array-like` or `Tensor`
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be a scalar.

            If ``index`` is a slice or a sequence of slices, ``value``
            must be broadcastable to the shape of the slice.
        """
        raise NotImplementedError('abstract method')

    __eq__ = LinearSpaceElement.__eq__

    def __ne__(self, other):
        """Return ``self != other``."""
        return not self == other

    @property
    def space(self):
        """The space to which this tensor belongs."""
        return self.__space

    @property
    def shape(self):
        """Number of elements per axis."""
        return self.space.shape

    @property
    def dtype(self):
        """Data type of each entry."""
        return self.space.dtype

    @property
    def order(self):
        """Data storage order, either ``'K'``, ``'C'`` or ``'F'``."""
        return self.space.order

    @property
    def view_order(self):
        """Order argument for view-preserving operations.

        This is identical to `order` except for ``self.order == 'K'``,
        where ``'A'`` is returned.
        """
        return self.space.view_order

    @property
    def size(self):
        """Total number of entries."""
        return self.space.size

    @property
    def ndim(self):
        """Number of axes (=dimensions) of this tensor."""
        return self.space.ndim

    def __len__(self):
        """Return ``len(self)``.

        The length is equal to the number of entries along axis 0.
        """
        return len(self.space)

    @property
    def itemsize(self):
        """Size in bytes of one tensor entry."""
        return self.space.itemsize

    @property
    def nbytes(self):
        """Total number of bytes in memory occupied by this tensor."""
        return self.space.nbytes

    def __array__(self, dtype=None):
        """Return a Numpy array from this tensor.

        Parameters
        ----------
        dtype :
            Specifier for the data type of the output array.

        Returns
        -------
        array : `numpy.ndarray`
        """
        if dtype is None:
            return self.asarray()
        else:
            return self.asarray().astype(dtype, copy=False)

    def __array_wrap__(self, obj):
        """Return a new tensor wrapping the array ``obj``.

        Parameters
        ----------
        obj : `numpy.ndarray`
            Array to be wrapped.

        Returns
        -------
        wrapper : `Tensor`
            Tensor wrapping ``obj``.
        """
        if obj.ndim == 0:
            return self.space.field.element(obj)
        else:
            return self.space.element(obj)

    def __str__(self):
        """Return ``str(self)``."""
        return arraynd_str(self)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            inner_str = arraynd_repr(self)
        else:
            inner_str = '\n' + indent_rows(arraynd_repr(self)) + '\n'
        return '{!r}.element({})'.format(self.space, inner_str)

    @property
    def ufuncs(self):
        """`TensorSpaceUfuncs`, access to Numpy style ufuncs.

        These are always available, but may or may not be optimized for
        the specific space in use.
        """
        return TensorSpaceUfuncs(self)

    def show(self, title=None, method='scatter', force_show=False, fig=None,
             **kwargs):
        """Display this tensor graphically for ``ndim == 1 or 2``.

        Parameters
        ----------
        title : str, optional
            Set the title of the figure

        method : str, optional
            1d methods:

            'plot' : graph plot

            'scatter' : point plot

        force_show : bool, optional
            Whether the plot should be forced to be shown now or deferred
            until later. Note that some backends always display the plot,
            regardless of this value.

        fig : `matplotlib.figure.Figure`
            The figure to show in. Expected to be of same "style", as
            the figure given by this function. The most common use case
            is that ``fig`` is the return value from an earlier call to
            this function.

        kwargs : {'figsize', 'saveto', ...}
            Extra keyword arguments passed on to display method
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
        # TODO: take code from DiscreteLpVector and adapt
        from odl.util.graphics import show_discrete_data
        from odl.discr import uniform_grid
        grid = uniform_grid(0, self.size - 1, self.size)
        return show_discrete_data(self.asarray(), grid, title=title,
                                  method=method, force_show=force_show,
                                  fig=fig, **kwargs)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
