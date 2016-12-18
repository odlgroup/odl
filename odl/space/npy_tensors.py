# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""CPU implementations of tensor spaces using NumPy."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
from future.utils import native
standard_library.install_aliases()
from builtins import super

import ctypes
from functools import partial
from math import sqrt
import numpy as np
import scipy.sparse
import scipy.linalg as linalg

from odl.operator.operator import Operator
from odl.set.sets import RealNumbers, ComplexNumbers
from odl.space.base_tensors import (
    TensorSet, GeneralizedTensor, TensorSpace, Tensor)
from odl.space.weighting import (
    Weighting, ArrayWeighting, ConstWeighting, NoWeighting,
    CustomInner, CustomNorm, CustomDist)
from odl.util.ufuncs import NumpyTensorSetUfuncs
from odl.util.utility import (
    dtype_str, signature_string,
    is_real_dtype, is_real_floating_dtype, is_complex_floating_dtype)


__all__ = ('NumpyTensorSet', 'NumpyTensorSpace')


_BLAS_DTYPES = (np.dtype('float32'), np.dtype('float64'),
                np.dtype('complex64'), np.dtype('complex128'))


class NumpyTensorSet(TensorSet):

    """The set of tensors of arbitrary type."""

    def __init__(self, shape, dtype, order='C'):
        """Initialize a new instance.

        Parameters
        ----------
        shape : sequence of int
            Number of elements per axis.
        dtype :
            Data type of each element. Can be provided in any
            way the `numpy.dtype` function understands, e.g.
            as built-in type or as a string.
        order : {'C', 'F'}, optional
            Axis ordering of the data storage.
        """
        TensorSet.__init__(self, shape, dtype, order)

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : `array-like`, optional
            Input to initialize the new element with.

            If ``inp`` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).

            If ``inp`` is a `numpy.ndarray` of the same shape and data
            type as this space, the array is wrapped, not copied.
            Other array-like objects are copied.

        Returns
        -------
        element : `NumpyGeneralizedTensor`
            The new element created (from ``inp``).

        Notes
        -----
        This method preserves "array views" of correct size and type,
        see the examples below.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x
        rn((2, 3)).element(
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
        )
        >>> x.space
        rn((2, 3))

        Construction from data pointer:

        >>> int_tset = odl.tensor_set((2, 3), dtype='int')
        >>> x = int_tset.element([[1, 2, 3],
        ...                       [4, 5, 6]])
        >>> y = int_tset.element(data_ptr=x.data_ptr)
        >>> print(y)
        [[1, 2, 3],
         [4, 5, 6]]
        >>> y[0, 1] = -1
        >>> print(x)
        [[1, -1, 3],
         [4, 5, 6]]
        """
        # TODO: exactly the same code would work for NumpyNtuples, factor out!
        if inp is None:
            if data_ptr is None:
                arr = np.empty(self.shape, dtype=self.dtype, order=self.order)
                return self.element_type(self, arr)
            else:
                ctype_array_def = ctypes.c_byte * self.nbytes
                as_ctype_array = ctype_array_def.from_address(data_ptr)
                as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
                arr = as_numpy_array.view(dtype=self.dtype).reshape(
                    self.shape, order=self.order)
                return self.element_type(self, arr)
        else:
            if data_ptr is None:
                if inp in self:
                    return inp
                else:
                    arr = np.array(inp, copy=False, dtype=self.dtype, ndmin=1,
                                   order='A').reshape(self.shape)

                    return self.element_type(self, arr)
            else:
                raise ValueError('cannot provide both `inp` and `data_ptr`')

    def zero(self):
        """Return a tensor of all zeros.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.zero()
        >>> x
        rn((2, 3)).element(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0]]
        )
        """
        return self.element(np.zeros(self.shape, dtype=self.dtype,
                                     order=self.order))

    def one(self):
        """Return a tensor of all ones.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.one()
        >>> x
        rn((2, 3)).element(
        [[1.0, 1.0, 1.0],
         [1.0, 1.0, 1.0]]
        )
        """
        return self.element(np.ones(self.shape, dtype=self.dtype,
                                    order=self.order))

    @staticmethod
    def default_order():
        """Return the default axis ordering of this implementation."""
        return 'C'

    @staticmethod
    def available_dtypes():
        """Return the list of data types available in this implementation.

        Notes
        -----
        This is all dtypes available in Numpy. See `numpy.sctypes`
        for more information.

        The available dtypes may depend on the specific system used.
        """
        all_types = []
        for val in np.sctypes.values():
            all_types.extend(val)
        return all_types

    @property
    def element_type(self):
        """Type of elements in this space: `NumpyGeneralizedTensor`."""
        return NumpyGeneralizedTensor

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            posargs = [self.size]
        else:
            posargs = [self.shape]
        constructor_name = 'tensor_set'
        posargs.append(dtype_str(self.dtype))
        optargs = [('order', self.order, self.default_order())]
        return '{}({})'.format(constructor_name,
                               signature_string(posargs, optargs))


class NumpyGeneralizedTensor(GeneralizedTensor):

    """Representation of a `NumpyTensorSet` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        if not isinstance(space, NumpyTensorSet):
            raise TypeError('`space` must be a `NumpyTensorSet` '
                            'instance, got {!r}'.format(space))

        if not isinstance(data, np.ndarray):
            raise TypeError('`data` {!r} not a `numpy.ndarray` instance'
                            ''.format(data))

        if data.dtype != space.dtype:
            raise TypeError('`data` {!r} not of dtype {!r}'
                            ''.format(data, space.dtype))
        self._data = data

        GeneralizedTensor.__init__(self, space)

    @property
    def data(self):
        """The `numpy.ndarray` representing the data of ``self``."""
        return self._data

    def asarray(self, out=None):
        """Extract the data of this array as a ``numpy.ndarray``.

        Parameters
        ----------
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.

        Returns
        -------
        asarray : `numpy.ndarray`
            Numpy array with the same data type as ``self``. If
            ``out`` was given, the returned object is a reference
            to it.

        Examples
        --------
        >>> space = odl.rn((2, 3), dtype='float32')
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x.asarray()
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)

        Using the out parameter:

        >>> out = np.empty((2, 3), dtype='float32')
        >>> result = x.asarray(out=out)
        >>> out
        array([[ 1.,  2.,  3.],
               [ 4.,  5.,  6.]], dtype=float32)

        >>> result is out
        True
        """
        if out is None:
            return self.data
        else:
            out[:] = self.data
            return out

    @property
    def data_ptr(self):
        """A raw pointer to the data container of ``self``.

        Examples
        --------
        >>> import ctypes
        >>> tset = odl.tensor_set((2, 3), dtype='int32')
        >>> x = tset.element([[1, 2, 3],
        ...                   [4, 5, 6]])
        >>> arr_type = ctypes.c_int32 * 6  # C type "array of 6 int32"
        >>> buffer = arr_type.from_address(x.data_ptr)
        >>> arr = np.frombuffer(buffer, dtype='int32')
        >>> print(arr)
        [1 2 3 4 5 6]

        In-place modification via pointer:

        >>> arr[0] = -1
        >>> print(x)
        [[-1, 2, 3],
         [4, 5, 6]]
        """
        return self.data.ctypes.data

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if all entries of ``other`` are equal to this
            the entries of ``self``, False otherwise.

        Notes
        -----
        Space membership is not checked, hence vectors from
        different spaces can be equal.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> y = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x == y
        True

        >>> y = space.element([[-1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x == y
        False

        Space membership matters:

        >>> space2 = odl.tensor_space((2, 3), dtype='int64')
        >>> y = space2.element([[1, 2, 3],
        ...                     [4, 5, 6]])
        >>> x == y or y == x
        False
        """
        if other is self:
            return True
        elif other not in self.space:
            return False
        else:
            return np.array_equal(self.data, other.data)

    def copy(self):
        """Create an identical (deep) copy of this vector.

        Parameters
        ----------
        None

        Returns
        -------
        copy : `NumpyGeneralizedTensor`
            The deep copy

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> y = x.copy()
        >>> y == x
        True
        >>> y is x
        False
        """
        return self.space.element(self.data.copy())

    __copy__ = copy

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be accessed.

        Returns
        -------
        values : `NumpyTensorSet.dtype` or `NumpyGeneralizedTensor`
            The value(s) at the given indices. Note that the returned
            object is a writable view into the original tensor.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x[0, 1]
        2.0
        >>> x[:, 1:]
        rn((2, 2)).element(
        [[2.0, 3.0],
         [5.0, 6.0]]
        )
        """
        arr = self.data[indices]
        if np.isscalar(arr):
            return arr
        else:
            return type(self.space)(arr.shape, dtype=self.dtype,
                                    order=self.order).element(arr)

    def __setitem__(self, indices, values):
        """Implement ``self[indices] = values``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be written to.
        values : scalar, array-like or `NumpyGeneralizedTensor`
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be a scalar.

            If ``index`` is a slice or a sequence of slices, ``value``
            must be broadcastable to the shape of the slice.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 2, 3],
        ...                    [4, 5, 6]])
        >>> x[0, 1] = -1
        >>> x
        rn((2, 3)).element(
        [[1.0, -1.0, 3.0],
         [4.0, 5.0, 6.0]]
        )

        Assignment from array-like structures or another tensor
        (may be in a different space):

        >>> tset = odl.tensor_set((2, 2), 'short')
        >>> y = tset.element([[-1, 2],
        ...                   [0, 0]])
        >>> x[:, :2] = y
        >>> x
        rn((2, 3)).element(
        [[-1.0, 2.0, 3.0],
         [0.0, 0.0, 6.0]]
        )
        >>> x[0, 1:] = [7, 8]
        >>> x
        rn((2, 3)).element(
        [[-1.0, 7.0, 8.0],
         [0.0, 0.0, 6.0]]
        )
        >>> import numpy as np
        >>> x[:] = np.array([[0, 0, 0],
        ...                  [1, 1, 1]])
        >>> x
        rn((2, 3)).element(
        [[0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0]]
        )

        Broadcasting is also supported:

        >>> x[:, ::2] = -2.
        >>> x
        rn((2, 3)).element(
        [[-2.0, 0.0, -2.0],
         [-2.0, 1.0, -2.0]]
        )

        Array views are preserved:

        >>> y = x[:, ::2]  # view into x
        >>> y[:] = -9
        >>> y
        rn((2, 2)).element(
        [[-9.0, -9.0],
         [-9.0, -9.0]]
        )
        >>> x
        rn((2, 3)).element(
        [[-9.0, 0.0, -9.0],
         [-9.0, 1.0, -9.0]]
        )

        Be aware of unsafe casts and over-/underflows, there
        will be warnings at maximum.

        >>> tset = odl.tensor_set((2, 3), 'int8')
        >>> x = tset.element([[0, 0, 0],
        ...                   [1, 1, 1]])
        >>> maxval = 255  # maximum signed 8-bit unsigned int
        >>> x[0, :] = maxval + 1
        >>> x
        tensor_set((2, 3), 'int8').element(
        [[0, 0, 0],
         [1, 1, 1]]
        )
        """
        if isinstance(values, NumpyGeneralizedTensor):
            self.data[indices] = values.data
        else:
            self.data[indices] = values

    @property
    def ufuncs(self):
        """`NumpyTensorSetUfuncs`, access to numpy style ufuncs.

        Notes
        -----
        These ufuncs are optimized for use with `NumpyGeneralizedTensor`'s
        and incur practically no overhead.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, -2, 3],
        ...                    [4, -5, 6]])
        >>> x.ufuncs.absolute()
        rn((2, 3)).element(
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
        )

        Broadcasting and array-like input is supported, too:

        >>> x.ufuncs.add(3)
        rn((2, 3)).element(
        [[4.0, 1.0, 6.0],
         [7.0, -2.0, 9.0]]
        )
        >>> x.ufuncs.subtract([[0, 0, 1],
        ...                   [1, 0, 0]])
        rn((2, 3)).element(
        [[1.0, -2.0, 2.0],
         [3.0, -5.0, 6.0]]
        )

        There is also support for various reductions (sum, prod, min,
        max):

        >>> x.ufuncs.sum()
        7.0

        The ``out`` parameter can be used for result storage:

        >>> y = space.one()
        >>> out = space.element()
        >>> result = x.ufuncs.add(y, out=out)
        >>> result
        rn((2, 3)).element(
        [[2.0, -1.0, 4.0],
         [5.0, -4.0, 7.0]]
        )
        >>> result is out
        True
        """
        return NumpyTensorSetUfuncs(self)


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.

    Parameters
    ----------
    x1,...,xN : `NumpyGeneralizedTensor`
        The tensors to be tested for BLAS conformity.
    """
    return (all(x.dtype == args[0].dtype and
                x.dtype in _BLAS_DTYPES and
                x.data.flags.contiguous
                for x in args))


def _lincomb(a, x1, b, x2, out, dtype):
    """Raw linear combination depending on data type."""

    # Shortcut for small problems
    if x1.size < 100:  # small array optimization
        out.data[:] = a * x1.data + b * x2.data
        return

    # Use blas for larger problems
    def fallback_axpy(x1, x2, n, a):
        """Fallback axpy implementation avoiding copy."""
        if a != 0:
            x2 /= a
            x2 += x1
            x2 *= a
        return x2

    def fallback_scal(a, x, n):
        """Fallback scal implementation."""
        x *= a
        return x

    def fallback_copy(x1, x2, n):
        """Fallback copy implementation."""
        x2[...] = x1[...]
        return x2

    if _blas_is_applicable(x1, x2, out):
        axpy, scal, copy = linalg.blas.get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x1.data, x2.data, out.data))
    else:
        axpy, scal, copy = (fallback_axpy, fallback_scal, fallback_copy)

    if x1 is x2 and b != 0:
        # x1 is aligned with x2 -> out = (a+b)*x1
        _lincomb(a + b, x1, 0, x1, out, dtype)
    elif out is x1 and out is x2:
        # All the vectors are aligned -> out = (a+b)*out
        scal(a + b, out.data, native(out.size))
    elif out is x1:
        # out is aligned with x1 -> out = a*out + b*x2
        if a != 1:
            scal(a, out.data, native(out.size))
        if b != 0:
            axpy(x2.data, out.data, native(out.size), b)
    elif out is x2:
        # out is aligned with x2 -> out = a*x1 + b*out
        if b != 1:
            scal(b, out.data, native(out.size))
        if a != 0:
            axpy(x1.data, out.data, native(out.size), a)
    else:
        # We have exhausted all alignment options, so x1 != x2 != out
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> out = 0
                out.data[:] = 0
            else:  # Scaled copy -> out = a*x1
                copy(x1.data, out.data, native(out.size))
                if a != 1:
                    scal(a, out.data, native(out.size))
        else:
            if a == 0:  # Scaled copy -> out = b*x2
                copy(x2.data, out.data, native(out.size))
                if b != 1:
                    scal(b, out.data, native(out.size))

            elif a == 1:  # No scaling in x1 -> out = x1 + b*x2
                copy(x1.data, out.data, native(out.size))
                axpy(x2.data, out.data, native(out.size), b)
            else:  # Generic case -> out = a*x1 + b*x2
                copy(x2.data, out.data, native(out.size))
                if b != 1:
                    scal(b, out.data, native(out.size))
                axpy(x1.data, out.data, native(out.size), a)


class NumpyTensorSpace(TensorSpace, NumpyTensorSet):

    """The space of tensors of given shape.

    This space implements multi-dimensional arrays whose entries are
    elements of a `Field`, which is usually the real or complex numbers.

    The space elements are represented as instances of the
    `NumpyTensor` class.
    """

    def __init__(self, shape, dtype=None, order='C', **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        shape : positive int or sequence of positive int
            Number of elements per axis of elements in this space.
        dtype : optional
            The data type of the storage array. For ``None``,
            the default data type of this space is used.
            Only scalar data types are allowed.

            ``dtype`` can be given in any way the `numpy.dtype`
            function understands, e.g. as built-in type or as a string.

        order : {'C', 'F'}, optional
            Axis ordering of the data storage.

        exponent : positive float, optional
            Exponent of the norm. For values other than 2.0, no
            inner product is defined.

            This option is ignored if ``dist``, ``norm`` or
            ``inner`` is given.

            Default: 2.0

        Other Parameters
        ----------------
        weighting : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weighting``:

            None: no weighting (default)

            `Weighting`: Use this weighting as-is. Compatibility
            with this space's elements is not checked during init.

            float: Weighting by a constant

            array-like: Pointwise weighting by an array

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``.

        dist : callable, optional
            Distance function defining a metric on the space.
            It must accept two `NumpyTensor` arguments and return
            a non-negative real number. See ``Notes`` for
            mathematical requirements.

            By default, ``dist(x, y)`` is calculated as ``norm(x - y)``.
            This creates an intermediate array ``x - y``, which can be
            avoided by choosing ``dist_using_inner=True``.

            This option cannot be combined with ``weight``,
            ``norm`` or ``inner``.

        norm : callable, optional
            The norm implementation. It must accept a
            `NumpyTensor` argument, return a non-negative real number.
            See ``Notes`` for mathematical requirements.

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            This option cannot be combined with ``weight``,
            ``dist`` or ``inner``.

        inner : callable, optional
            The inner product implementation. It must accept two
            `NumpyTensor` arguments and return an element of the field
            of the space (usually real or complex number).
            See ``Notes`` for mathematical requirements.

            This option cannot be combined with ``weight``,
            ``dist`` or ``norm``.

        dist_using_inner : bool, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.

            Default: False.

        kwargs :
            Further keyword arguments are passed to the weighting
            classes.

        See also
        --------
        rn : constructor for real tensor spaces
        cn : constructor for complex tensor spaces
        tensor_space :
            constructor for tensor spaces of arbitrary scalar
            data type


        Notes
        -----
        - A distance function or metric on a space :math:`\mathcal{X}`
          is a mapping
          :math:`d:\mathcal{X} \\times \mathcal{X} \\to \mathbb{R}`
          satisfying the following conditions for all space elements
          :math:`x, y, z`:

          * :math:`d(x, y) \geq 0`,
          * :math:`d(x, y) = 0 \Leftrightarrow x = y`,
          * :math:`d(x, y) = d(y, x)`,
          * :math:`d(x, y) \\leq d(x, z) + d(z, y)`.

        - A norm on a space :math:`\mathcal{X}` is a mapping
          :math:`\| \cdot \|:\mathcal{X} \\to \mathbb{R}`
          satisfying the following conditions for all
          space elements :math:`x, y`: and scalars :math:`s`:

          * :math:`\| x\| \geq 0`,
          * :math:`\| x\| = 0 \Leftrightarrow x = 0`,
          * :math:`\| sx\| = |s| \cdot \| x \|`,
          * :math:`\| x+y\| \\leq \| x\| +
            \| y\|`.

        - An inner product on a space :math:`\mathcal{X}` over a field
          :math:`\mathbb{F} = \mathbb{R}` or :math:`\mathbb{C}` is a
          mapping
          :math:`\\langle\cdot, \cdot\\rangle: \mathcal{X} \\times
          \mathcal{X} \\to \mathbb{F}`
          satisfying the following conditions for all
          space elements :math:`x, y, z`: and scalars :math:`s`:

          * :math:`\\langle x, y\\rangle =
            \overline{\\langle y, x\\rangle}`,
          * :math:`\\langle sx + y, z\\rangle = s \\langle x, z\\rangle +
            \\langle y, z\\rangle`,
          * :math:`\\langle x, x\\rangle = 0 \Leftrightarrow x = 0`.

        Examples
        --------
        >>> tspace = odl.tensor_space((2, 3), dtype='float64')
        >>> tspace
        rn((2, 3))

        >>> tspace = odl.tensor_space((2, 3), dtype='complex64')
        >>> tspace
        cn((2, 3), 'complex64')

        >>> tspace = odl.tensor_space((2, 3), dtype='int64')
        >>> tspace
        tensor_space((2, 3), 'int')
        """
        NumpyTensorSet.__init__(self, shape, dtype, order)
        TensorSpace.__init__(self, shape, dtype, order)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weighting = kwargs.pop('weighting', None)
        exponent = kwargs.pop('exponent', 2.0)
        dist_using_inner = bool(kwargs.pop('dist_using_inner', False))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weighting)) < 3:
            raise ValueError('invalid combination of options `weighting`, '
                             '`dist`, `norm` and `inner`')

        if any(x is not None for x in (dist, norm, inner)) and exponent != 2.0:
            raise ValueError('`exponent` cannot be used together with '
                             '`dist`, `norm` and `inner`')

        # Set the weighting
        if weighting is not None:
            if isinstance(weighting, Weighting):
                self.__weighting = weighting
            else:
                self.__weighting = _weighting(
                    weighting, exponent, dist_using_inner=dist_using_inner)

            # Check (afterwards) that the weighting input was sane
            if isinstance(self.weighting, NumpyTensorSpaceArrayWeighting):
                if self.weighting.array.dtype == object:
                    raise ValueError('invalid `weighting` argument: {}'
                                     ''.format(weighting))
                elif not np.can_cast(self.weighting.array.dtype, self.dtype):
                    raise ValueError(
                        'cannot cast from `weighting` data type {} to '
                        'the space `dtype` {}'
                        ''.format(dtype_str(self.weighting.array.dtype),
                                  dtype_str(self.dtype)))
                if self.weighting.array.shape != self.shape:
                    raise ValueError('array-like weights must have same '
                                     'shape {} as this space, got {}'
                                     ''.format(self.shape,
                                               self.weighting.array.shape))

        elif dist is not None:
            self.__weighting = NumpyTensorSpaceCustomDist(dist)
        elif norm is not None:
            self.__weighting = NumpyTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self.__weighting = NumpyTensorSpaceCustomInner(inner)
        else:
            self.__weighting = NumpyTensorSpaceNoWeighting(
                exponent, dist_using_inner=dist_using_inner)

    @property
    def exponent(self):
        """Exponent of the norm and the distance."""
        return self.weighting.exponent

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self.__weighting

    @property
    def is_weighted(self):
        """Return True if the space has a non-trivial weighting."""
        return not isinstance(self.weighting, NumpyTensorSpaceNoWeighting)

    @staticmethod
    def available_dtypes():
        """Return the list of data types available in this implementation.

        Notes
        -----
        This is the set of all arithmetic dtypes available in Numpy. See
        `numpy.sctypes` for more information.

        The available dtypes may depend on the specific system used.
        """
        return np.sctypes['int'] + np.sctypes['float'] + np.sctypes['complex']

    @staticmethod
    def default_dtype(field=None):
        """Return the default data type of this class for a given field.

        Parameters
        ----------
        field : `Field`, optional
            Set of numbers to be represented by a data type.
            Currently supported : `RealNumbers`, `ComplexNumbers`
            The default ``None`` means `RealNumbers`

        Returns
        -------
        dtype : `numpy.dtype`
            Numpy data type specifier. The returned defaults are:

                ``RealNumbers()`` : ``np.dtype('float64')``

                ``ComplexNumbers()`` : ``np.dtype('complex128')``
        """
        if field is None or field == RealNumbers():
            return np.dtype('float64')
        elif field == ComplexNumbers():
            return np.dtype('complex128')
        else:
            raise ValueError('no default data type defined for field {}'
                             ''.format(field))

    zero = NumpyTensorSet.zero
    one = NumpyTensorSet.one

    def _lincomb(self, a, x1, b, x2, out):
        """Implement the linear combination of ``x1`` and ``x2``.

        Compute ``out = a*x1 + b*x2`` using optimized
        BLAS routines if possible.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        a, b : `TensorSpace.field` element
            Scalars to multiply ``x1`` and ``x2`` with.
        x1, x2 : `NumpyTensor`
            Summands in the linear combination.
        out : `NumpyTensor`
            Tensor to which the result is written.

        Examples
        --------
        >>> space = odl.cn((2, 3))
        >>> x = space.element([[0, 1j, 1],
        ...                    [0, 1, 0]])
        >>> y = space.element([[0, 0, 1j],
        ...                    [-1j, 0, 2j]])
        >>> out = space.element()
        >>> space.lincomb(1j, x, 1, y, out)  # out is returned
        cn((2, 3)).element(
        [[0j, (-1+0j), 2j],
         [-1j, 1j, 2j]]
        )
        >>> out
        cn((2, 3)).element(
        [[0j, (-1+0j), 2j],
         [-1j, 1j, 2j]]
        )
        """
        _lincomb(a, x1, b, x2, out, self.dtype)

    def _dist(self, x1, x2):
        """Return the distance between ``x1`` and ``x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Elements whose mutual distance is calculated.

        Returns
        -------
        dist : `float`
            Distance between the elements.

        Examples
        --------
        >>> space_2 = odl.rn((2, 3), exponent=2)
        >>> x = space_2.element([[-1, 1, 2],
        ...                      [1, -1, 1]])
        >>> y = space_2.one()
        >>> space_2.dist(x, y)
        3.0

        >>> space_1 = odl.rn((2, 3), exponent=1)
        >>> x = space_1.element([[-1, 1, 2],
        ...                      [1, -1, 1]])
        >>> y = space_1.one()
        >>> space_1.dist(x, y)
        5.0

        Weighting is supported, too:

        >>> weights = [[2, 1, 1], [1, 1, 2]]
        >>> space_1_w = odl.rn((2, 3), exponent=1, weighting=weights)
        >>> x = space_1_w.element([[-1, 1, 2],
        ...                        [1, -1, 1]])
        >>> y = space_1_w.one()
        >>> space_1_w.dist(x, y)
        7.0
        """
        return self.weighting.dist(x1, x2)

    def _norm(self, x):
        """Return the norm of ``x``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x : `NumpyTensor`
            Element whose norm is calculated.

        Returns
        -------
        norm : `float`
            Norm of the element.

        Examples
        --------
        >>> space_2 = odl.rn((2, 3), exponent=2)
        >>> x = space_2.element([[1, 0, 3],
        ...                      [4, -1, 3]])
        >>> space_2.norm(x)
        6.0
        >>> space_1 = odl.rn((2, 3), exponent=1)
        >>> x = space_1.element([[1, 0, 3],
        ...                      [4, -1, 3]])
        >>> space_1.norm(x)
        12.0

        Weighting is supported:

        >>> weights = [[1, 2, 1], [1, 1, 2]]
        >>> space_1_w = odl.rn((2, 3), exponent=1,
        ...                    weighting=weights)
        >>> x = space_1_w.element([[1, 0, 3],
        ...                        [4, -1, 3]])
        >>> space_1_w.norm(x)
        15.0
        """
        return self.weighting.norm(x)

    def _inner(self, x1, x2):
        """Return the inner product of ``x1`` and ``x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Elements whose inner product is calculated.

        Returns
        -------
        inner : `field` `element`
            Inner product of the elements.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 0, 3],
        ...                    [4, -1, 3]])
        >>> y = space.one()
        >>> space.inner(x, y)
        10.0

        Weighting is supported, too:

        >>> weights = [[1, 2, 1],
        ...            [2, 1, 1]]
        >>> space_w = odl.rn((2, 3), weighting=weights)
        >>> x = space_w.element([[1, 0, 3],
        ...                      [4, -1, 3]])
        >>> y = space_w.one()
        >>> space_w.inner(x, y)
        14.0
        """
        return self.weighting.inner(x1, x2)

    def _multiply(self, x1, x2, out):
        """Compute the entry-wise product ``out = x1 * x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Factors in the product.
        out : `NumpyTensor`
            Element to which the result is written.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[1, 0, 3],
        ...                    [4, -1, 3]])
        >>> y = space.element([[-1, 1, -1],
        ...                    [1, -1, 1]])
        >>> out = space.element()
        >>> space.multiply(x, y, out=out)  # out is returned
        rn((2, 3)).element(
        [[-1.0, 0.0, -3.0],
         [4.0, 1.0, 3.0]]
        )
        >>> out
        rn((2, 3)).element(
        [[-1.0, 0.0, -3.0],
         [4.0, 1.0, 3.0]]
        )
        """
        np.multiply(x1.data, x2.data, out=out.data)

    def _divide(self, x1, x2, out):
        """Compute the entry-wise quotient ``x1 / x2``.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Dividend and divisor in the quotient.
        out : `NumpyTensor`
            Element to which the result is written.

        Examples
        --------
        >>> space = odl.rn((2, 3))
        >>> x = space.element([[2, 0, 4],
        ...                    [-4, 0, 2]])
        >>> y = space.element([[1, 1, 2],
        ...                    [-4, 1, 2]])
        >>> out = space.element()
        >>> space.divide(x, y, out=out)  # out is returned
        rn((2, 3)).element(
        [[2.0, 0.0, 2.0],
         [1.0, 0.0, 1.0]]
        )
        >>> out
        rn((2, 3)).element(
        [[2.0, 0.0, 2.0],
         [1.0, 0.0, 1.0]]
        )
        """
        np.divide(x1.data, x2.data, out=out.data)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : bool
            True if ``other`` is an instance of ``type(self)``
            with the same `NumpyTensorSet.shape`,
            `NumpyTensorSet.dtype`, `NumpyTensorSet.order`
            and `NumpyTensorSet.weighting`, otherwise False.

        Examples
        --------
        >>> import odl
        >>> tspace = odl.rn((2, 3))
        >>> same_tspace = odl.rn((2, 3), exponent=2)
        >>> same_tspace == tspace
        True

        Different `shape`, `exponent` or `dtype` all result in
        different spaces:

        >>> diff_tspace = odl.rn((2, 3, 4))
        >>> diff_tspace == tspace
        False
        >>> diff_tspace = odl.rn((2, 3), exponent=1)
        >>> diff_tspace == tspace
        False
        >>> diff_tspace = odl.rn((2, 3), dtype='float32')
        >>> diff_tspace == tspace
        False

        A `NumpyTensorSpace` with the same properties is considered
        equal:

        >>> same_tspace = odl.NumpyTensorSpace((2, 3), dtype='float64')
        >>> same_tspace == tspace
        True
        """
        if other is self:
            return True

        return (NumpyTensorSet.__eq__(self, other) and
                self.weighting == other.weighting)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.ndim == 1:
            posargs = [self.size]
        else:
            posargs = [self.shape]

        if self.is_real_space:
            constructor_name = 'rn'
        elif self.is_complex_space:
            constructor_name = 'cn'
        else:
            constructor_name = 'tensor_space'

        if (constructor_name == 'tensor_space' or
                self.dtype != self.default_dtype(self.field)):
            posargs.append(dtype_str(self.dtype))

        optargs = [('order', self.order, self.default_order())]
        inner_str = signature_string(posargs, optargs)
        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str

        return '{}({})'.format(constructor_name, inner_str)

    @property
    def element_type(self):
        """Type of elements in this space: `NumpyTensor`."""
        return NumpyTensor


class NumpyTensor(Tensor, NumpyGeneralizedTensor):

    """Representation of a `NumpyTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        if not isinstance(space, NumpyTensorSpace):
            raise TypeError('`space` must be a `NumpyTensorSpace` instance, '
                            'got {!r}'.format(space))

        Tensor.__init__(self, space)
        NumpyGeneralizedTensor.__init__(self, space, data)

    @property
    def real(self):
        """Real part of ``self``.

        Returns
        -------
        real : `NumpyTensor`
            Real part of this element as a member of a
            `NumpyTensorSpace` with corresponding real data type.

        Examples
        --------
        >>> import odl
        >>> tspace = odl.cn((2, 3))
        >>> x = tspace.element([[1 + 1j, 2, 3 - 3j],
        ...                     [4, 5 - 5j, 6]])
        >>> x.real
        rn((2, 3)).element(
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
        )
        """
        real_space = self.space.astype(self.space.real_dtype)
        return real_space.element(self.data.real)

    @real.setter
    def real(self, newreal):
        """Setter for the real part.

        This method is invoked by ``x.real = other``.

        Parameters
        ----------
        newreal : array-like or scalar
            Values to be assigned to the real part of this element.

        Examples
        --------
        >>> import odl
        >>> tspace = odl.cn((2, 3))
        >>> x = tspace.element([[1 + 1j, 2, 3 - 3j],
        ...                     [4, 5 - 5j, 6]])
        >>> zero = odl.rn((2, 3)).zero()
        >>> x.real = zero
        >>> x
        cn((2, 3)).element(
        [[1j, 0j, -3j],
         [0j, -5j, 0j]]
        )

        Other array-like types and broadcasting:

        >>> x.real = 1.0
        >>> x
        cn((2, 3)).element(
        [[(1+1j), (1+0j), (1-3j)],
         [(1+0j), (1-5j), (1+0j)]]
        )
        >>> x.real = [[2, 3, 4],
        ...           [5, 6, 7]]
        >>> x
        cn((2, 3)).element(
        [[(2+1j), (3+0j), (4-3j)],
         [(5+0j), (6-5j), (7+0j)]]
        )
        """
        self.real.data[:] = newreal

    @property
    def imag(self):
        """Imaginary part of ``self``.

        Returns
        -------
        imag : `NumpyTensor`
            Imaginary part this element as an element of a
            `NumpyTensorSpace` with real data type.

        Examples
        --------
        >>> import odl
        >>> tspace = odl.cn((2, 3))
        >>> x = tspace.element([[1 + 1j, 2, 3 - 3j],
        ...                     [4, 5 - 5j, 6]])
        >>> x.imag
        rn((2, 3)).element(
        [[1.0, 0.0, -3.0],
         [0.0, -5.0, 0.0]]
        )
        """
        real_space = self.space.astype(self.space.real_dtype)
        return real_space.element(self.data.imag)

    @imag.setter
    def imag(self, newimag):
        """Setter for the imaginary part.

        This method is invoked by ``x.imag = other``.

        Parameters
        ----------
        newimag : array-like or scalar
            Values to be assigned to the imaginary part of this element.

        Examples
        --------
        >>> import odl
        >>> tspace = odl.cn((2, 3))
        >>> x = tspace.element([[1 + 1j, 2, 3 - 3j],
        ...                     [4, 5 - 5j, 6]])
        >>> zero = odl.rn((2, 3)).zero()
        >>> x.imag = zero
        >>> x
        cn((2, 3)).element(
        [[(1+0j), (2+0j), (3+0j)],
         [(4+0j), (5+0j), (6+0j)]]
        )

        Other array-like types and broadcasting:

        >>> x.imag = 1.0
        >>> x
        cn((2, 3)).element(
        [[(1+1j), (2+1j), (3+1j)],
         [(4+1j), (5+1j), (6+1j)]]
        )
        >>> x.imag = [[2, 3, 4],
        ...           [5, 6, 7]]
        >>> x
        cn((2, 3)).element(
        [[(1+2j), (2+3j), (3+4j)],
         [(4+5j), (5+6j), (6+7j)]]
        )
        """
        self.imag.data[:] = newimag

    def conj(self, out=None):
        """Return the complex conjugate of ``self``.

        Parameters
        ----------
        out : `NumpyTensor`, optional
            Element to which the complex conjugate is written.
            Must be an element of ``self.space``.

        Returns
        -------
        out : `NumpyTensor`
            The complex conjugate element. If ``out`` was provided,
            the returned object is a reference to it.

        Examples
        --------
        >>> import odl
        >>> tspace = odl.cn((2, 3))
        >>> x = tspace.element([[1 + 1j, 2, 3 - 3j],
        ...                     [4, 5 - 5j, 6]])
        >>> x.conj()
        cn((2, 3)).element(
        [[(1-1j), (2-0j), (3+3j)],
         [(4-0j), (5+5j), (6-0j)]]
        )

        The out parameter allows you to avoid a copy

        >>> y = tspace.element()
        >>> y_out = x.conj(out=y)
        >>> y
        cn((2, 3)).element(
        [[(1-1j), (2-0j), (3+3j)],
         [(4-0j), (5+5j), (6-0j)]]
        )
        >>> y_out is y
        True

        It can also be used for in-place conjugation:

        >>> x_out = x.conj(out=x)
        >>> x
        cn((2, 3)).element(
        [[(1-1j), (2-0j), (3+3j)],
         [(4-0j), (5+5j), (6-0j)]]
        )
        """
        if out is None:
            return self.space.element(self.data.conj())
        else:
            self.data.conj(out.data)
            return out

    def __getitem__(self, indices):
        """Return ``self[indices]``.

        Parameters
        ----------
        indices : index expression
            Integer, slice or sequence of these, defining the positions
            of the data array which should be accessed.

        Returns
        -------
        values : `NumpyTensorSet.dtype` or `NumpyGeneralizedTensor`
            The value(s) at the given indices. Note that the returned
            object is a writable view into the original tensor.

        Examples
        --------
        >>> import odl
        >>> tspace = odl.rn((2, 3))
        >>> x = tspace.element([[1, 2, 3],
        ...                     [4, 5, 6]])
        >>> x[0, 1]
        2.0
        >>> x[:, 1:]
        rn((2, 2)).element(
        [[2.0, 3.0],
         [5.0, 6.0]]
        )
        """
        arr = self.data[indices]
        if np.isscalar(arr):
            return arr
        else:
            space_constructor = type(self.space)
            space = space_constructor(
                arr.shape, dtype=self.dtype, order=self.order,
                exponent=self.space.exponent, weighting=self.space.weighting)
            return space.element(arr)

    def __ipow__(self, other):
        """Return ``self **= other``."""
        try:
            if other == int(other):
                return super().__ipow__(other)
        except TypeError:
            pass

        np.power(self.data, other, out=self.data)
        return self

    def __int__(self):
        """Return ``int(self)``.

        Raises
        ------
        TypeError
            if ``self.size != 1``.
        """
        return int(self.data)

    def __long__(self):
        """Return ``long(self)``.

        This method is only available in Python 2.

        Raises
        ------
        TypeError
            if ``self.size != 1``.
        """
        return long(self.data)

    def __float__(self):
        """Return ``float(self)``.

        Raises
        ------
        TypeError
            if ``self.size != 1``.
        """
        return float(self.data)

    def __complex__(self):
        """Return ``complex(self)``.

        Raises
        ------
        TypeError
            if ``self.size != 1``.
        """
        if self.size != 1:
            raise TypeError('only size-1 tensors can be converted to '
                            'Python scalars')
        return complex(self.data.ravel()[0])


def _weighting(weights, exponent, dist_using_inner=False):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weights):
        weighting = NumpyTensorSpaceConstWeighting(
            weights, exponent, dist_using_inner=dist_using_inner)
    elif weights is None:
        weighting = NumpyTensorSpaceNoWeighting(
            exponent, dist_using_inner=dist_using_inner)
    else:  # last possibility: make an array
        arr = np.asarray(weights)
        weighting = NumpyTensorSpaceArrayWeighting(
            arr, exponent, dist_using_inner=dist_using_inner)
    return weighting


def npy_weighted_inner(weights):
    """Weighted inner product on `Fn` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the inner product. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector.

    Returns
    -------
    inner : `callable`
        Inner product function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    NumpyTensorSpaceConstWeighting
    NumpyTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=2.0).inner


def npy_weighted_norm(weights, exponent=2.0):
    """Weighted norm on `Fn` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the norm. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector.
    exponent : positive `float`
        Exponent of the norm.

    Returns
    -------
    norm : `callable`
        Norm function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    NumpyTensorSpaceConstWeighting
    NumpyTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent).norm


def npy_weighted_dist(weights, exponent=2.0, use_inner=False):
    """Weighted distance on `Fn` spaces as free function.

    Parameters
    ----------
    weights : scalar or `array-like`
        Weights of the distance. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector.
    exponent : positive `float`
        Exponent of the norm.
    use_inner : `bool`, optional
        Calculate ``dist`` using the formula

            ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

        This avoids the creation of new arrays and is thus faster
        for large arrays. On the downside, it will not evaluate to
        exactly zero for equal (but not identical) ``x`` and ``y``.

        Can only be used if ``exponent`` is 2.0.

    Returns
    -------
    dist : `callable`
        Distance function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    NumpyTensorSpaceConstWeighting
    NumpyTensorSpaceArrayWeighting
    """
    return _weighting(weights, exponent=exponent,
                      dist_using_inner=use_inner).dist


def _norm_default(x):
    """Default Euclidean norm implementation."""
    if _blas_is_applicable(x):
        nrm2 = linalg.blas.get_blas_funcs('nrm2', dtype=x.dtype)
        norm = partial(nrm2, n=native(x.size))
    else:
        norm = np.linalg.norm
    return norm(x.data)


def _pnorm_default(x, p):
    """Default p-norm implementation."""
    return np.linalg.norm(x.data.ravel(), ord=p)


def _pnorm_diagweight(x, p, w):
    """Diagonally weighted p-norm implementation."""
    # This is faster than first applying the weights and then summing with
    # BLAS dot or nrm2
    xp = np.abs(x.data.ravel())
    if np.isfinite(p):
        xp = np.power(xp, p, out=xp)
        xp *= w.ravel()  # w is a plain NumPy array
        return np.sum(xp) ** (1 / p)
    else:
        xp *= w.ravel()
        return np.max(xp)


def _inner_default(x1, x2):
    """Default Euclidean inner product implementation."""
    if _blas_is_applicable(x1, x2):
        dotc = linalg.blas.get_blas_funcs('dotc', dtype=x1.dtype)
        dot = partial(dotc, n=native(x1.size))
    elif is_real_dtype(x1.dtype):
        dot = np.dot  # still much faster than vdot
    else:
        dot = np.vdot  # slowest alternative
    # x2 as first argument because we want linearity in x1
    return dot(x2.data.ravel(), x1.data.ravel())


# TODO: implement intermediate weighting schemes with arrays that are
# broadcast, i.e. between scalar and full-blown in dimensionality?


class NumpyTensorSpaceArrayWeighting(ArrayWeighting):

    """Weighting of a `NumpyTensorSpace` by an array.

    This class defines a weighting by an array that has the same shape
    as the tensor space. Since the space is not known to this class,
    no checks of shape or data type are performed.
    See ``Notes`` for mathematical details.
    """

    def __init__(self, array, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        array : `array-like`, one-dim.
            Weighting array of the inner product, norm and distance.
            All its entries must be positive, however this is not
            verified during initialization.
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, no inner
            product is defined.
        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with array
          :math:`W` is defined as

          .. math::
              \\langle A, B\\rangle_W :=
              \\langle W \odot A, B\\rangle =
              \\langle w \odot a, b\\rangle =
              b^{\mathrm{H}} (w \odot a),

          where :math:`a, b, w` are the "flattened" counterparts of
          tensors :math:`A, B, W`, respectively, :math:`b^{\mathrm{H}}`
          stands for transposed complex conjugate and :math:`w \odot a`
          for element-wise multiplication.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\\infty`, the weighted norm is

          .. math::
              \| A\|_{W, \\infty} :=
              \| W \odot A\|_{\\infty} =
              \| w \odot a\|_{\\infty},

          otherwise it is (using point-wise exponentiation)

          .. math::
              \| A\|_{W, p} :=
              \| W^{1/p} \odot A\|_{p} =
              \| w^{1/p} \odot a\|_{\\infty}.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

          .. math::
              \| A\|_{W, p} \\not\\to
              \| A\|_{W, \\infty} \quad (p \\to \\infty)

          unless all weights are equal to 1.

        - The array :math:`W` may only have positive entries, otherwise
          it does not define an inner product or norm, respectively. This
          is not checked during initialization.
        """
        super().__init__(array, impl='numpy', exponent=exponent,
                         dist_using_inner=dist_using_inner)

    def inner(self, x1, x2):
        """Return the weighted inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Tensors whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            inner = _inner_default(x1 * self.array, x2)
            if is_real_dtype(x1.dtype):
                return float(inner)
            else:
                return complex(inner)

    def norm(self, x):
        """Return the weighted norm of ``x``.

        Parameters
        ----------
        x : `NumpyTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the provided tensor.
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?!
            if norm_squared < 0:
                norm_squared = 0.0  # Compensate for numerical error
            return sqrt(norm_squared)
        else:
            return float(_pnorm_diagweight(x, self.exponent, self.array))


class NumpyTensorSpaceConstWeighting(ConstWeighting):

    """Weighting of a `NumpyTensorSpace` by a constant.

    See ``Notes`` for mathematical details.
    """

    def __init__(self, const, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        const : positive float
            Weighting constant of the inner product, norm and distance.
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.

        Notes
        -----
        - For exponent 2.0, a new weighted inner product with constant
          :math:`c` is defined as

          .. math::
              \\langle a, b\\rangle_c :=
              c \, \\langle a, b\\rangle_c =
              c \, b^{\mathrm{H}} a,

          where :math:`b^{\mathrm{H}}` standing for transposed complex
          conjugate.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\\infty`, the weighted norm is defined
          as

          .. math::
              \| a \|_{c, \\infty} :=
              c\, \| a \|_{\\infty},

          otherwise it is

          .. math::
              \| a \|_{c, p} :=
              c^{1/p}\, \| a \|_{p}.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

          .. math::
              \| a\|_{c, p} \\not\\to
              \| a \|_{c, \\infty} \quad (p \\to \\infty)

          unless :math:`c = 1`.

        - The constant must be positive, otherwise it does not define an
          inner product or norm, respectively.
        """
        super().__init__(const, impl='numpy', exponent=exponent,
                         dist_using_inner=dist_using_inner)

    def inner(self, x1, x2):
        """Return the weighted inner product of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Tensors whose inner product is calculated.

        Returns
        -------
        inner : float or complex
            The inner product of the two provided tensors.
        """
        if self.exponent != 2.0:
            raise NotImplementedError('no inner product defined for '
                                      'exponent != 2 (got {})'
                                      ''.format(self.exponent))
        else:
            inner = self.const * _inner_default(x1, x2)
            return x1.space.field.element(inner)

    def norm(self, x):
        """Return the weighted norm of ``x``.

        Parameters
        ----------
        x1 : `NumpyTensor`
            Tensor whose norm is calculated.

        Returns
        -------
        norm : float
            The norm of the tensor.
        """
        if self.exponent == 2.0:
            return sqrt(self.const) * float(_norm_default(x))
        elif self.exponent == float('inf'):
            return self.const * float(_pnorm_default(x, self.exponent))
        else:
            return (self.const ** (1 / self.exponent) *
                    float(_pnorm_default(x, self.exponent)))

    def dist(self, x1, x2):
        """Return the weighted distance between ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : `NumpyTensor`
            Tensors whose mutual distance is calculated.

        Returns
        -------
        dist : float
            The distance between the tensors.
        """
        if self.dist_using_inner:
            dist_squared = (_norm_default(x1) ** 2 + _norm_default(x2) ** 2 -
                            2 * _inner_default(x1, x2).real)
            if dist_squared < 0.0:  # Compensate for numerical error
                dist_squared = 0.0
            return sqrt(self.const) * float(sqrt(dist_squared))
        elif self.exponent == 2.0:
            return sqrt(self.const) * _norm_default(x1 - x2)
        elif self.exponent == float('inf'):
            return self.const * float(_pnorm_default(x1 - x2, self.exponent))
        else:
            return (self.const ** (1 / self.exponent) *
                    float(_pnorm_default(x1 - x2, self.exponent)))


class NumpyTensorSpaceNoWeighting(NoWeighting,
                                  NumpyTensorSpaceConstWeighting):

    """Weighting of a `NumpyTensorSpace` with constant 1."""

    # Implement singleton pattern for efficiency in the default case
    __instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern if ``exp==2.0``."""
        if len(args) == 0:
            exponent = kwargs.pop('exponent', 2.0)
            dist_using_inner = kwargs.pop('dist_using_inner', False)
        elif len(args) == 1:
            exponent = args[0]
            args = args[1:]
            dist_using_inner = kwargs.pop('dist_using_inner', False)
        else:
            exponent = args[0]
            dist_using_inner = args[1]
            args = args[2:]

        if exponent == 2.0 and not dist_using_inner:
            if not cls.__instance:
                cls.__instance = super().__new__(cls, *args, **kwargs)
            return cls.__instance
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.

            This option can only be used if ``exponent`` is 2.0.
        """
        super().__init__(impl='numpy', exponent=exponent,
                         dist_using_inner=dist_using_inner)


class NumpyTensorSpaceCustomInner(CustomInner):

    """Class for handling a user-specified inner product."""

    def __init__(self, inner, dist_using_inner=True):
        """Initialize a new instance.

        Parameters
        ----------
        inner : `callable`
            The inner product implementation. It must accept two
            `FnVector` arguments, return an element from their space's
            field (real or complex number) and satisfy the following
            conditions for all vectors ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``

        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

                ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.
        """
        super().__init__(inner, impl='numpy',
                         dist_using_inner=dist_using_inner)


class NumpyTensorSpaceCustomNorm(CustomNorm):

    """Class for handling a user-specified norm.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : `callable`
            The norm implementation. It must accept an `FnVector`
            argument, return a `float` and satisfy the following
            conditions for all vectors ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super().__init__(norm, impl='numpy')


class NumpyTensorSpaceCustomDist(CustomDist):

    """Class for handling a user-specified distance in `Fn`.

    Note that this removes ``inner`` and ``norm``.
    """

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : `callable`
            The distance function defining a metric on `Fn`. It must
            accept two `FnVector` arguments, return a `float` and and
            fulfill the following mathematical conditions for any three
            vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
        """
        super().__init__(dist, impl='numpy')


class MatrixOperator(Operator):

    """A matrix acting as a linear operator.

    This operator uses a matrix to represent an operator, and get its
    adjoint and inverse by doing computations on the matrix. This is in
    general a rather slow and memory-inefficient approach, and users are
    recommended to use other alternatives if possible.
    """

    def __init__(self, matrix, domain=None, range=None, axis=0):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : `array-like` or  `scipy.sparse.base.spmatrix`
            2-dimensional array representing the linear operator.
        domain : `NumpyTensorSpace`, optional
            Space of elements on which the operator can act. Its
            ``dtype`` must be castable to ``range.dtype``.
            For the default ``None``, a space with 1 axis and size
            ``matrix.shape[1]`` is used, together with the matrix'
            data type.
        range : `NumpyTensorSpace`, optional
            Space of elements on to which the operator maps. Its
            ``shape`` and ``dtype`` attributes must match the ones
            of the result of the multiplication.
            For the default ``None``, the range is inferred from
            ``matrix``, ``domain`` and ``axis``.
        axis : int, optional
            Sum over this axis of an input tensor in the
            multiplication.

        Examples
        --------
        >>> m = np.ones((3, 4))
        >>>
        TODO: implement example

        Notes
        -----
        For a matrix :math:`A \\in \\mathbb{F}^{n \\times m}`, the
        operation on a tensor :math:`T \\in \mathbb{F}^{n_1 \\times
        \dots \\times n_d}` is defined as the summation

        .. math::
            (A \cdot T)_{i_1, \dots, i_k, \dots, i_d} =
            \sum_{j=1}^m A_{i_k j} T_{i_1, \dots, j, \dots, i_d}.

        It produces a new tensor :math:`A \cdot T \in \mathbb{F}^{
        n_1 \\times \dots \\times n \\times \dots \\times n_d}`.
        """
        if scipy.sparse.isspmatrix(matrix):
            self.__matrix = matrix
        else:
            self.__matrix = np.asarray(matrix)

        self.__axis, axis_in = int(axis), axis
        if self.axis != axis_in:
            raise ValueError('`axis` must be integer, got {}'.format(axis_in))

        if self.matrix.ndim != 2:
            raise ValueError('`matrix` has {} axes instead of 2'
                             ''.format(self.matrix.ndim))

        # Infer or check domain
        if domain is None:
            domain = NumpyTensorSpace((self.matrix.shape[1],),
                                      dtype=self.matrix.dtype)
        else:
            if not isinstance(domain, NumpyTensorSpace):
                raise TypeError('`domain` must be not a `NumpyTensorSpace`'
                                'instance, got {!r}'.format(domain))

            if scipy.sparse.isspmatrix(self.matrix) and domain.ndim > 1:
                raise ValueError('`domain.ndim` > 1 unsupported for '
                                 'scipy sparse matrices')

            if domain.shape[axis] != self.matrix.shape[1]:
                raise ValueError('`domain.shape[axis]` not equal to '
                                 '`matrix.shape[1]` ({} != {})'
                                 ''.format(domain.shape[axis],
                                           self.matrix.shape[1]))

        # Infer or check range
        range_shape = list(domain.shape)
        range_shape[self.axis] = self.matrix.shape[0]

        if range is None:
            range_dtype = np.promote_types(self.matrix.dtype, domain.dtype)
            range = NumpyTensorSpace(range_shape, dtype=range_dtype)
        else:
            if not isinstance(range, NumpyTensorSpace):
                raise TypeError('`range` must be not a `NumpyTensorSpace`'
                                'instance, got {!r}'.format(range))

            if range.shape != tuple(range_shape):
                raise ValueError('expected `range.shape` = {}, got {}'
                                 ''.format(tuple(range_shape), range.shape))

        # Check compatibility of data types
        if not np.can_cast(domain.dtype, range.dtype):
            raise TypeError('domain data type {!r} cannot be safely cast to '
                            'range data type {!r}'
                            ''.format(domain.dtype, range.dtype))

        if not np.can_cast(self.matrix.dtype, range.dtype):
            raise TypeError('matrix data type {!r} cannot be safely cast to '
                            'range data type {!r}.'
                            ''.format(matrix.dtype, range.dtype))

        super().__init__(domain, range, linear=True)

    @property
    def matrix(self):
        """Matrix representing this operator."""
        return self.__matrix

    @property
    def axis(self):
        """Axis of domain elements over which is summed."""
        return self.__axis

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix.

        Returns
        -------
        adjoint : `MatrixOperator`
        """
        return MatrixOperator(self.matrix.conj().T,
                              domain=self.range, range=self.domain,
                              axis=self.axis)

    @property
    def inverse(self):
        """Inverse operator represented by the inverse matrix.

        Taking the inverse causes sparse matrices to become dense and is
        generally very heavy computationally since the matrix is inverted
        numerically (an O(n^3) operation). It is recommended to instead
        use one of the solvers available in the ``odl.solvers`` package.

        Returns
        -------
        inverse : `MatrixOperator`
        """
        if self.matrix_issparse:
            dense_matrix = self.matrix.toarray()
        else:
            dense_matrix = self.matrix

        return MatrixOperator(np.linalg.inv(dense_matrix),
                              domain=self.range, range=self.domain,
                              axis=self.axis)

    def _call(self, x, out=None):
        """Raw apply method on input, writing to given output."""
        if out is None:
            if scipy.sparse.isspmatrix(self.matrix):
                out = self.matrix.dot(x.data)
            else:
                out = np.tensordot(self.matrix, x.data, axes=(1, self.axis))

            return out

        else:
            if scipy.sparse.isspmatrix(self.matrix):
                # Unfortunately, there is no native in-place dot product for
                # sparse matrices
                out.data[:] = self.matrix.dot(x.data)
            elif self.range.ndim == 1:
                self.matrix.dot(x.data, out=out.data)
            else:
                # Could use einsum to have out, but it's damn slow
                out.data[:] = np.tensordot(self.matrix, x.data,
                                           axes=(1, self.axis))

    def __repr__(self):
        """Return ``repr(self)``."""
        sig_str = '{!r}'.format(self.matrix)
        opt_arg_strings = []
        if self.domain != NumpyTensorSpace((self.matrix.shape[1],),
                                           dtype=self.matrix.dtype):
            opt_arg_strings.append('{!r}'.format(self.domain))
        range_shape = list(self.domain.shape)
        range_shape[self.axis] = self.matrix.shape[0]
        if self.range != NumpyTensorSpace(range_shape,
                                          dtype=self.matrix.dtype):
            opt_arg_strings.append('{!r}'.format(self.domain))

        if opt_arg_strings:
            sig_str += ',\n' + ', '.join(opt_arg_strings)

        return '{}(\n{}\n)'.format(self.__class__.__name__, sig_str)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
