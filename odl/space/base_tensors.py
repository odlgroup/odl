# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Base classes for implementations of tensor spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from numbers import Integral
import numpy as np

from odl.set.sets import RealNumbers, ComplexNumbers
from odl.set.space import LinearSpace, LinearSpaceElement
from odl.util import (
    is_numeric_dtype, is_real_dtype,
    is_real_floating_dtype, is_complex_floating_dtype, safe_int_conv,
    array_str, dtype_str, signature_string, indent, writable_array)
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

    See the `Wikipedia article on tensors`_ for further details.
    See also [Hac2012]_ "Part I Algebraic Tensors" for a rigorous
    treatment of tensors with a definition close to this one.

    References
    ----------
    [Hac2012] Hackbusch, W. *Tensor Spaces and Numerical Tensor Calculus*.
    Springer, 2012.

    .. _Wikipedia article on tensors: https://en.wikipedia.org/wiki/Tensor
    """

    def __init__(self, shape, dtype, order='A'):
        """Initialize a new instance.

        Parameters
        ----------
        shape : nonnegative int or sequence of nonnegative ints
            Number of entries per axis for elements in this space. A
            single integer results in a space with rank 1, i.e., 1 axis.
        dtype :
            Scalar data type of elements in this space. Can be provided
            in any way the `numpy.dtype` constructor understands, e.g.
            as built-in type or as a string.
        order : {'A', 'C', 'F'}, optional
            Axis ordering of the data storage. Only relevant for more
            than 1 axis.
            For ``'C'`` and ``'F'``, elements are forced to use
            contiguous memory in the respective ordering.
            For ``'A'`` ("any") no contiguousness is enforced.
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

        self.__order = str(order).upper()
        if self.order not in ('A', 'C', 'F'):
            raise ValueError("`order {!r} not understood".format(order))

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
    def is_real_space(self):
        """True if this is a space of real tensors."""
        return is_real_floating_dtype(self.dtype)

    @property
    def is_complex_space(self):
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
        NotImplementedError
            If `dtype` is not a numeric data type.
        """
        if not is_numeric_dtype(self.dtype):
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
        if not is_numeric_dtype(self.dtype):
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
        order : {'A', 'C', 'F'}, optional
            Axis ordering of the data storage. Only relevant for more
            than 1 axis.
            For ``'C'`` and ``'F'``, elements are forced to use
            contiguous memory in the respective ordering.
            For ``'A'`` ("any") no contiguousness is enforced.
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

        if is_numeric_dtype(self.dtype):
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

        This is one of ``('A', 'C', 'F')``, where ``'A'`` means "any",
        i.e. "no guarantee".
        """
        return self.__order

    @property
    def default_order(self):
        """Storage order for new elements in this space.

        This is identical to `order` except for ``self.order == 'A'``,
        where ``'C'`` is returned.
        """
        return 'C' if self.order == 'A' else self.order

    @property
    def size(self):
        """Total number of entries in an element of this space."""
        return 0 if self.shape == () else int(np.prod(self.shape))

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
        """Total number of bytes used by an element of this space in memory."""
        return self.size * self.itemsize

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        This base version of indexing does not propagate weightings.
        Subclasses need to override this method if weighting propagation
        is desired.

        Examples
        --------
        Integers and slices index the first axis:

        >>> space = odl.rn((2, 3, 4))
        >>> space[1]
        rn((3, 4))
        >>> space[1:]
        rn((1, 3, 4))

        Tuples, i.e. multi-indices, index per axis:

        >>> space[0, 1]
        rn(4)
        >>> space[0, 1:]
        rn((2, 4))
        """
        if isinstance(indices, Integral):
            newshape = self.shape[1:]

        elif isinstance(indices, slice):
            # Take slice along axis 0
            start, stop, step = indices.indices(self.shape[0])
            newshape = (int(np.ceil((stop - start) / step)),) + self.shape[1:]

        elif isinstance(indices, list):
            # Fancy indexing, only for compatibility since nothing really
            # interesting happens here
            idx_arr = np.array(indices, dtype=int, ndmin=2)
            for i, (row, length) in enumerate(zip(idx_arr, self.shape)):
                row[row < 0] += length
                if np.any((row < 0) | (row >= length)):
                    raise IndexError('list contains invalid indices in row {}'
                                     ''.format(i))
            newshape = (idx_arr.shape[1],) + self.shape[idx_arr.shape[0]:]

        elif isinstance(indices, tuple):
            newshape = []
            for ax, (idx, length) in enumerate(zip(indices, self.shape)):
                if isinstance(idx, Integral):
                    norm_idx = length - idx if idx < 0 else idx
                    if not 0 <= norm_idx < length:
                        raise IndexError(
                            'index {} out of range in axis {} with length {}'
                            ''.format(idx, ax, length))
                elif isinstance(idx, slice):
                    start, stop, step = idx.indices(length)
                    if start >= length:
                        newshape.append(0)
                    else:
                        stop = min(stop, length)
                        newshape.append(int(np.ceil((stop - start) / step)))
                else:
                    raise TypeError('index tuple may only contain'
                                    'integers or slices')
            newshape.extend(self.shape[len(indices):])
            newshape = tuple(newshape)

        else:
            raise TypeError('`indices` must be integer, slice, tuple or '
                            'or list, got {!r}'.format(indices))

        return type(self)(newshape, self.dtype, self.order)

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

        return (type(other) is type(self) and
                self.shape == other.shape and
                self.dtype == other.dtype and
                self.order == other.order)

    def __hash__(self):
        """Return ``hash(self)``."""
        return hash((type(self), self.shape, self.dtype, self.order))

    def __repr__(self):
        """Return ``repr(self)``."""
        posargs = [self.shape, dtype_str(self.dtype)]
        optargs = [('order', self.order, 'A')]
        return "{}({})".format(self.__class__.__name__,
                               signature_string(posargs, optargs))

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
        """Return the set of data types available in this implementation."""
        raise NotImplementedError('abstract method')

    @property
    def element_type(self):
        """Type of elements in this set: `Tensor`."""
        return Tensor


class Tensor(LinearSpaceElement):

    """Abstract class for representation of `TensorSpace` elements."""

    def asarray(self, out=None):
        """Extract the data of this tensor as a Numpy array.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
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

    @property
    def impl(self):
        """Name of the implementation back-end of this tensor."""
        return self.space.impl

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
        """Data storage order, either ``'A'``, ``'C'`` or ``'F'``."""
        return self.space.order

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

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            inner_str = array_str(self)
        else:
            inner_str = '\n' + indent(array_str(self)) + '\n'
        return '{!r}.element({})'.format(self.space, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        return array_str(self)

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

    def __array_wrap__(self, array):
        """Return a new tensor wrapping the ``array``.

        Parameters
        ----------
        array : `numpy.ndarray`
            Array to be wrapped.

        Returns
        -------
        wrapper : `Tensor`
            Tensor wrapping ``array``.
        """
        if array.ndim == 0:
            return self.space.field.element(array)
        else:
            return self.space.element(array)

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
            This basic implementation casts inputs and
            outputs to Numpy arrays and evaluates ``ufunc`` on those.
            For `numpy.ndarray` based data storage, this incurs no
            significant overhead compared to direct usage of Numpy arrays.

            For other (in particular non-local) implementations, e.g.,
            GPU arrays or distributed memory, overhead is significant due
            to copies to CPU main memory. In those classes, the
            ``__array_ufunc__`` mechanism should be overridden to use
            native implementations if possible.

        .. note::
            When using operations that alter the shape (like ``reduce``),
            or the data type (can be any of the methods),
            the resulting array is wrapped in a space of the same
            type as ``self.space``, however **only** using the minimal
            set of parameters ``shape``, ``dtype`` and ``order``. If more
            properties (e.g. weighting) should be propagated, this method
            must be overridden.

        Parameters
        ----------
        ufunc : `numpy.ufunc`
            Ufunc that should be called on ``self``.
        method : str
            Method on ``ufunc`` that should be called on ``self``.
            Possible values:

            ``'__call__'``, ``'accumulate'``, ``'at'``, ``'outer'``,
            ``'reduce'``, ``'reduceat'``

        input1, ..., inputN:
            Positional arguments to ``ufunc.method``.
        kwargs:
            Keyword arguments to ``ufunc.method``.

        Returns
        -------
        ufunc_result : `Tensor`, `numpy.ndarray` or tuple
            Result of the ufunc evaluation. If no ``out`` keyword argument
            was given, the result is a `Tensor` or a tuple
            of such, depending on the number of outputs of ``ufunc``.
            If ``out`` was provided, the returned object or tuple entries
            refer(s) to ``out``.

        Examples
        --------
        We apply `numpy.add` to ODL tensors:

        >>> r3 = odl.rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([-1, -2, -3])
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn(3).element([ 0.,  0.,  0.])
        >>> np.add(x, y)  # same mechanism for Numpy >= 1.13
        rn(3).element([ 0.,  0.,  0.])

        As ``out``, a Numpy array or an ODL tensor can be given (wrapped
        in a sequence):

        >>> out = r3.element()
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out,))
        >>> out
        rn(3).element([ 0.,  0.,  0.])
        >>> res is out
        True
        >>> out_arr = np.empty(3)
        >>> res = x.__array_ufunc__(np.add, '__call__', x, y, out=(out_arr,))
        >>> out_arr
        array([ 0.,  0.,  0.])
        >>> res is out_arr
        True

        With multiple dimensions:

        >>> r23 = odl.rn((2, 3))
        >>> x = y = r23.one()
        >>> x.__array_ufunc__(np.add, '__call__', x, y)
        rn((2, 3)).element(
            [[ 2.,  2.,  2.],
             [ 2.,  2.,  2.]]
        )

        The ``ufunc.accumulate`` method retains the original `shape` and
        `dtype`. The latter can be changed with the ``dtype`` parameter:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'accumulate', x)
        rn(3).element([ 1.,  3.,  6.])
        >>> np.add.accumulate(x)  # same mechanism for Numpy >= 1.13
        rn(3).element([ 1.,  3.,  6.])
        >>> x.__array_ufunc__(np.add, 'accumulate', x, dtype=complex)
        cn(3).element([ 1.+0.j,  3.+0.j,  6.+0.j])

        For multi-dimensional tensors, an optional ``axis`` parameter
        can be provided:

        >>> z = r23.one()
        >>> z.__array_ufunc__(np.add, 'accumulate', z, axis=1)
        rn((2, 3)).element(
            [[ 1.,  2.,  3.],
             [ 1.,  2.,  3.]]
        )

        The ``ufunc.at`` method operates in-place. Here we add the second
        operand ``[5, 10]`` to ``x`` at indices ``[0, 2]``:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'at', x, [0, 2], [5, 10])
        >>> x
        rn(3).element([  6.,   2.,  13.])

        For outer-product-type operations, i.e., operations where the result
        shape is the sum of the individual shapes, the ``ufunc.outer``
        method can be used:

        >>> x = odl.rn(2).element([0, 3])
        >>> y = odl.rn(3).element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'outer', x, y)
        rn((2, 3)).element(
            [[ 1.,  2.,  3.],
             [ 4.,  5.,  6.]]
        )
        >>> y.__array_ufunc__(np.add, 'outer', y, x)
        rn((3, 2)).element(
            [[ 1.,  4.],
             [ 2.,  5.],
             [ 3.,  6.]]
        )

        Using ``ufunc.reduce`` produces a scalar, which can be avoided with
        ``keepdims=True``:

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'reduce', x)
        6.0
        >>> x.__array_ufunc__(np.add, 'reduce', x, keepdims=True)
        rn(1).element([ 6.])

        In multiple dimensions, ``axis`` can be provided for reduction over
        selected axes:

        >>> z = r23.element([[1, 2, 3],
        ...                  [4, 5, 6]])
        >>> z.__array_ufunc__(np.add, 'reduce', z, axis=1)
        rn(2).element([  6.,  15.])

        Finally, ``add.reduceat`` is a combination of ``reduce`` and
        ``at`` with rather flexible and complex semantics (see the
        `reduceat documentation`_ for details):

        >>> x = r3.element([1, 2, 3])
        >>> x.__array_ufunc__(np.add, 'reduceat', x, [0, 1])
        rn(2).element([ 1.,  5.])

        References
        ----------
        .. _corresponding NEP:
           https://github.com/numpy/numpy/blob/master/doc/neps/\
ufunc-overrides.rst

        .. _interface documentation:
           https://github.com/charris/numpy/blob/master/doc/source/reference/\
arrays.classes.rst#special-attributes-and-methods

        .. _general documentation on Numpy ufuncs:
           https://docs.scipy.org/doc/numpy/reference/ufuncs.html

        .. _reduceat documentation:
           https://docs.scipy.org/doc/numpy/reference/generated/\
numpy.ufunc.reduceat.html
        """
        # Unwrap out if provided. The output parameters are all wrapped
        # in one tuple, even if there is only one.
        out_tuple = kwargs.pop('out', ())

        # We allow our own tensors and `numpy.ndarray` objects as `out`
        if not all(isinstance(out, (type(self), np.ndarray)) or out is None
                   for out in out_tuple):
            return NotImplemented

        # Convert inputs that are ODL tensors to arrays so that the
        # native Numpy ufunc is called later
        inputs = tuple(
            inp.asarray() if isinstance(inp, type(self)) else inp
            for inp in inputs)

        out = out1 = out2 = None
        if len(out_tuple) == 1:
            out = out_tuple[0]
        elif len(out_tuple) == 2:
            out1 = out_tuple[0]
            out2 = out_tuple[1]

        # Arguments for `writable_array` and/or space constructors
        order = str(kwargs.get('order', self.order)).upper()
        array_kwargs = {'order': order}
        order = 'A' if order == 'K' else order  # We don't use 'K'

        out_dtype = kwargs.get('dtype', None)
        if out_dtype is not None:
            array_kwargs['dtype'] = out_dtype

        # Trivial context to be used when an output is `None`
        class CtxNone(object):
            __enter__ = __exit__ = lambda *_: None

        if method == '__call__':
            if ufunc.nout == 1:
                # Make context for output (trivial one returns `None`)
                if out is None:
                    out_ctx = CtxNone()
                else:
                    out_ctx = writable_array(out, **array_kwargs)

                # Evaluate ufunc
                with out_ctx as out_arr:
                    kwargs['out'] = out_arr
                    res = ufunc(*inputs, **kwargs)

                # Wrap result if necessary (lazily)
                if out is None:
                    out_space = type(self.space)(self.shape, res.dtype, order)
                    out = out_space.element(res)

                return out

            elif ufunc.nout == 2:
                # Make contexts for outputs (trivial ones return `None`)
                if out1 is not None:
                    out1_ctx = writable_array(out1, **array_kwargs)
                else:
                    out1_ctx = CtxNone()
                if out2 is not None:
                    out2_ctx = writable_array(out2, **array_kwargs)
                else:
                    out2_ctx = CtxNone()

                # Evaluate ufunc
                with out1_ctx as out1_arr, out2_ctx as out2_arr:
                    kwargs['out'] = (out1_arr, out2_arr)
                    res1, res2 = ufunc(*inputs, **kwargs)

                # Wrap results if necessary (lazily)
                if out1 is None:
                    out1_space = type(self.space)(self.shape, res1.dtype,
                                                  order)
                    out1 = out1_space.element(res1)
                if out2 is None:
                    out2_space = type(self.space)(self.shape, res2.dtype,
                                                  order)
                    out2 = out2_space.element(res2)

                return out1, out2

            else:
                raise NotImplementedError('nout = {} not supported'
                                          ''.format(ufunc.nout))

        else:  # method != '__call__'
            # Make context for output (trivial one returns `None`)
            if out is None:
                out_ctx = CtxNone()
            else:
                out_ctx = writable_array(out, **array_kwargs)

            # Evaluate ufunc method
            with out_ctx as out_arr:
                kwargs['out'] = out_arr
                res = getattr(ufunc, method)(*inputs, **kwargs)

            # Shortcut for scalar or no return value
            if np.isscalar(res) or res is None:
                # The first occurs for `reduce` with all axes,
                # the second for in-place stuff (`at` currently)
                return res

            # Wrap result if necessary (lazily)
            if out is None:
                out_space = type(self.space)(res.shape, res.dtype, order)
                out = out_space.element(res)

            return out

    # Old ufuncs interface, will be deprecated when Numpy 1.13 becomes minimum

    @property
    def ufuncs(self):
        """Access to Numpy style universal functions.

        These default ufuncs are always available, but may or may not be
        optimized for the specific space in use.

        .. note::
            This interface is will be deprecated when Numpy 1.13 becomes
            the minimum required version. Use Numpy ufuncs directly, e.g.,
            ``np.sqrt(x)`` instead of ``x.ufuncs.sqrt()``.
        """
        return TensorSpaceUfuncs(self)

    def show(self, title=None, method='', indices=None, force_show=False,
             fig=None, **kwargs):
        """Display the function graphically.

        Parameters
        ----------
        title : string, optional
            Set the title of the figure

        method : string, optional
            1d methods:

                ``'plot'`` : graph plot

                ``'scatter'`` : scattered 2d points (2nd axis <-> value)

            2d methods:

                ``'imshow'`` : image plot with coloring according to
                value, including a colorbar.

                ``'scatter'`` : cloud of scattered 3d points
                (3rd axis <-> value)

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

        # Default to showing x-y slice "in the middle"
        if indices is None and self.ndim >= 3:
            indices = (slice(None),) * 2
            indices += tuple(n // 2 for n in self.space.shape[2:])

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
            raise ValueError('too few axes ({} < {})'.format(len(indices),
                                                             self.ndim))
        if len(indices) > self.ndim:
            raise ValueError('too many axes ({} > {})'.format(len(indices),
                                                              self.ndim))

        # Squeeze grid and values according to the index expression
        full_grid = uniform_grid([0] * self.ndim, np.array(self.shape) - 1,
                                 self.shape)
        grid = full_grid[indices].squeeze()
        values = self.asarray()[indices].squeeze()

        return show_discrete_data(values, grid, title=title, method=method,
                                  force_show=force_show, fig=fig, **kwargs)


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    run_doctests()
