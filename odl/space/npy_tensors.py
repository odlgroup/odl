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

"""CPU implementations of tensor spaces."""

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
import scipy.linalg as linalg

from odl.set.sets import RealNumbers, ComplexNumbers
from odl.space.base_ntuples import (
    NtuplesBase, NtuplesBaseVector, FnBase, FnBaseVector)
from odl.space.base_tensors import (
    TensorSetBase, GeneralTensorBase, TensorSpaceBase, TensorBase)
from odl.space.weighting import (
    WeightingBase, MatrixWeightingBase, ArrayWeightingBase,
    ConstWeightingBase, NoWeightingBase,
    CustomInnerBase, CustomNormBase, CustomDistBase)
from odl.util.ufuncs import NumpyGeneralTensorUFuncs
from odl.util.utility import (
    dtype_repr, is_real_dtype, is_real_floating_dtype,
    is_complex_floating_dtype)


__all__ = ('NumpyTensorSet', 'NumpyGeneralTensor',
           'NumpyTensorSpace', 'NumpyTensor')


_BLAS_DTYPES = (np.dtype('float32'), np.dtype('float64'),
                np.dtype('complex64'), np.dtype('complex128'))


class NumpyTensorSet(TensorSetBase):

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
        TensorSetBase.__init__(self, shape, dtype, order)

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
        element : `NumpyGeneralTensor`
            The new element created (from ``inp``).

        Notes
        -----
        This method preserves "array views" of correct size and type,
        see the examples below.

        Examples
        --------
        >>> strings3 = Ntuples(3, dtype='U1')  # 1-char strings
        >>> x = strings3.element(['w', 'b', 'w'])
        >>> print(x)

        >>> x.space


        Construction from data pointer:

        >>> int3 = Ntuples(3, dtype='int')
        >>> x = int3.element([1, 2, 3])
        >>> y = int3.element(data_ptr=x.data_ptr)
        >>> print(y)

        >>> y[0] = 5
        >>> print(x)

        """
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
                                   order='A')
                    if arr.shape != self.shape:
                        raise ValueError('expected input shape {}, got {}'
                                         ''.format((self.size,), arr.shape))

                    return self.element_type(self, arr)
            else:
                raise ValueError('cannot provide both `inp` and `data_ptr`')

    def zero(self):
        """Return a tensor of all zeros.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.zero()
        >>> x

        """
        return self.element(np.zeros(self.shape, dtype=self.dtype,
                                     order=self.order))

    def one(self):
        """Return a tensor of all ones.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.one()
        >>> x

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
        """Type of elements in this space: `NumpyGeneralTensor`."""
        return NumpyGeneralTensor


class NumpyGeneralTensor(GeneralTensorBase):

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

        GeneralTensorBase.__init__(self, space)

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
        >>> vec = Ntuples(3, 'float').element([1, 2, 3])
        >>> vec.asarray()

        >>> vec.asarray(start=1, stop=3)


        Using the out parameter:

        >>> out = np.empty((3,), dtype='float')
        >>> result = vec.asarray(out=out)
        >>> out

        >>> result is out

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
        >>> vec = Ntuples(3, 'int32').element([1, 2, 3])
        >>> arr_type = ctypes.c_int32 * 3
        >>> buffer = arr_type.from_address(vec.data_ptr)
        >>> arr = np.frombuffer(buffer, dtype='int32')
        >>> print(arr)


        In-place modification via pointer:

        >>> arr[0] = 5
        >>> print(vec)

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
        >>> vec1 = Ntuples(3, int).element([1, 2, 3])
        >>> vec2 = Ntuples(3, int).element([-1, 2, 0])
        >>> vec1 == vec2

        >>> vec2 = Ntuples(3, int).element([1, 2, 3])
        >>> vec1 == vec2


        Space membership matters:

        >>> vec2 = Ntuples(3, float).element([1, 2, 3])
        >>> vec1 == vec2 or vec2 == vec1

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
        copy : `NumpyGeneralTensor`
            The deep copy

        Examples
        --------
        >>> vec1 = Ntuples(3, 'int').element([1, 2, 3])
        >>> vec2 = vec1.copy()
        >>> vec2

        >>> vec1 == vec2

        >>> vec1 is vec2

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
        values : `NumpyTensorSet.dtype` or `NumpyGeneralTensor`
            The value(s) at the given indices. Note that the returned
            object is a writable view into the original tensor.

        Examples
        --------
        >>> str_3 = Ntuples(3, dtype='U6')  # 6-char unicode
        >>> x = str_3.element(['a', 'Hello!', '0'])
        >>> print(x[0])

        >>> print(x[1:3])

        >>> x[1:3].space

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
        values : scalar, array-like or `NumpyGeneralTensor`
            The value(s) that are to be assigned.

            If ``index`` is an integer, ``value`` must be a scalar.

            If ``index`` is a slice or a sequence of slices, ``value``
            must be broadcastable to the shape of the slice.

        Examples
        --------
        >>> int_3 = Ntuples(3, 'int')
        >>> x = int_3.element([1, 2, 3])
        >>> x[0] = 5
        >>> x


        Assignment from array-like structures or another
        vector:

        >>> y = Ntuples(2, 'short').element([-1, 2])
        >>> x[:2] = y
        >>> x

        >>> x[1:3] = [7, 8]
        >>> x

        >>> x[:] = np.array([0, 0, 0])
        >>> x


        Broadcasting is also supported:

        >>> x[1:3] = -2.
        >>> x


        Array views are preserved:

        >>> y = x[::2]  # view into x
        >>> y[:] = -9
        >>> print(y)

        >>> print(x)


        Be aware of unsafe casts and over-/underflows, there
        will be warnings at maximum.

        >>> x = Ntuples(2, 'int8').element([0, 0])
        >>> maxval = 255  # maximum signed 8-bit unsigned int
        >>> x[0] = maxval + 1
        >>> x

        """
        if isinstance(values, NumpyGeneralTensor):
            self.data[indices] = values.data
        else:
            self.data[indices] = values

    @property
    def ufunc(self):
        """`NtuplesUFuncs`, access to numpy style ufuncs.

        Examples
        --------
        >>> r2 = Rn(2)
        >>> x = r2.element([1, -2])
        >>> x.ufunc.absolute()


        These functions can also be used with broadcasting

        >>> x.ufunc.add(3)


        and non-space elements

        >>> x.ufunc.subtract([3, 3])


        There is also support for various reductions (sum, prod, min, max)

        >>> x.ufunc.sum()


        They also support an out parameter

        >>> y = r2.element([3, 4])
        >>> out = r2.element()
        >>> result = x.ufunc.add(y, out=out)
        >>> result

        >>> result is out


        Notes
        -----
        These are optimized for use with ntuples and incur no overhead.
        """
        return NumpyGeneralTensorUFuncs(self)


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.

    Parameters
    ----------
    x1,...,xN : `NumpyGeneralTensor`
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


class NumpyTensorSpace(TensorSpaceBase, NumpyTensorSet):

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
        size : positive int
            The number of dimensions of the space
        dtype : optional
            Data type of the storage array. For ``None``, the default
            data type of this space is used.
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

        weight : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weight``:

            None: no weighting (default)

            `WeightingBase`: Use this weighting as-is. Compatibility
            with this space's elements is not checked during init.

            float: Weighting by a constant

            array-like: Pointwise weighting by an array

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``.

        Other Parameters
        ----------------
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
        NumpyTensorSpaceArrayWeighting
        NumpyTensorSpaceConstWeighting

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
          :math:`\\lVert \cdot \\rVert:\mathcal{X} \\to \mathbb{R}`
          satisfying the following conditions for all
          space elements :math:`x, y`: and scalars :math:`s`:

          * :math:`\\lVert x\\rVert \geq 0`,
          * :math:`\\lVert x\\rVert = 0 \Leftrightarrow x = 0`,
          * :math:`\\lVert sx\\rVert = |s| \cdot \\lVert x \\rVert`,
          * :math:`\\lVert x+y\\rVert \\leq \\lVert x\\rVert +
            \\lVert y\\rVert`.

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
        >>> space = Fn(3, 'float')
        >>> space

        >>> space = Fn(3, 'float', weight=[1, 2, 3])
        >>> space

        """
        NumpyTensorSet.__init__(self, shape, dtype, order)
        TensorSpaceBase.__init__(self, shape, dtype, order)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)
        exponent = kwargs.pop('exponent', 2.0)
        dist_using_inner = bool(kwargs.pop('dist_using_inner', False))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weight)) < 3:
            raise ValueError('invalid combination of options `weight`, '
                             '`dist`, `norm` and `inner`')

        if any(x is not None for x in (dist, norm, inner)) and exponent != 2.0:
            raise ValueError('`exponent` cannot be used together with '
                             '`dist`, `norm` and `inner`')

        # Set the weighting
        if weight is not None:
            if isinstance(weight, WeightingBase):
                self._weighting = weight
            else:
                self._weighting = _weighting(weight, exponent,
                                             dist_using_inner=dist_using_inner)
            if isinstance(self.weighting, NumpyTensorSpaceArrayWeighting):
                if self.weighting.array.dtype == object:
                    raise ValueError('invalid weight argument {}'
                                     ''.format(weight))
                if self.weighting.array.ndim != self.ndim:
                    raise ValueError('array-like weight must have {} '
                                     'dimensions, got a {}-dim. array'
                                     ''.format(self.weighting.array.ndim))
                for i, (n_a, n_s) in enumerate(zip(self.weighting.array.shape,
                                                   self.shape)):
                    if n_a not in (1, n_s):
                        raise ValueError('in axis {}: expected shape 1 or '
                                         '{}, got {}'.format(i, n_s, n_a))

        elif dist is not None:
            self._weighting = NumpyTensorSpaceCustomDist(dist)
        elif norm is not None:
            self._weighting = NumpyTensorSpaceCustomNorm(norm)
        elif inner is not None:
            self._weighting = NumpyTensorSpaceCustomInner(inner)
        else:
            self._weighting = NumpyTensorSpaceNoWeighting(
                exponent, dist_using_inner=dist_using_inner)

    @property
    def exponent(self):
        """Exponent of the norm and the distance."""
        return self.weighting.exponent

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self._weighting

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
    def default_dtype(field):
        """Return the default data type of this class for a given field.

        Parameters
        ----------
        field : `Field`
            Set of numbers to be represented by a data type.
            Currently supported : `RealNumbers`, `ComplexNumbers`

        Returns
        -------
        dtype : `type`
            Numpy data type specifier. The returned defaults are:

            ``RealNumbers()`` : ``np.dtype('float64')``

            ``ComplexNumbers()`` : ``np.dtype('complex128')``
        """
        if field == RealNumbers():
            return np.dtype('float64')
        elif field == ComplexNumbers():
            return np.dtype('complex128')
        else:
            raise ValueError('no default data type defined for field {}'
                             ''.format(field))

    zero = NumpyTensorSet.zero
    one = NumpyTensorSet.one

    def _lincomb(self, a, x1, b, x2, out):
        """Calculate the linear combination of ``x1`` and ``x2``.

        Compute ``out = a*x1 + b*x2`` using optimized
        BLAS routines if possible.

        This function is part of the subclassing API. Do not
        call it directly.

        Parameters
        ----------
        a, b : `TensorSpaceBase.field` element
            Scalars to multiply ``x1`` and ``x2`` with.
        x1, x2 : `NumpyTensor`
            Summands in the linear combination.
        out : `NumpyTensor`
            Tensor to which the result is written.

        Returns
        -------
        `None`

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([1+1j, 2-1j, 3])
        >>> y = c3.element([4+0j, 5, 6+0.5j])
        >>> out = c3.element()
        >>> c3.lincomb(2j, x, 3-1j, y, out)  # out is returned

        >>> out

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
        >>> from numpy.linalg import norm
        >>> c2_2 = Cn(2, dist=lambda x, y: norm(x - y, ord=2))
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)


        >>> c2_2 = Cn(2, dist=lambda x, y: norm(x - y, ord=1))
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)

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
        >>> import numpy as np
        >>> c2_2 = Cn(2, norm=np.linalg.norm)  # 2-norm
        >>> x = c2_2.element([3+1j, 1-5j])
        >>> c2_2.norm(x)


        >>> from functools import partial
        >>> c2_1 = Cn(2, norm=partial(np.linalg.norm, ord=1))
        >>> x = c2_1.element([3-4j, 12+5j])
        >>> c2_1.norm(x)

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
        >>> import numpy as np
        >>> c3 = Cn(2, inner=lambda x, y: np.vdot(y, x))
        >>> x = c3.element([5+1j, -2j])
        >>> y = c3.element([1, 1+1j])
        >>> c3.inner(x, y) == (5+1j)*1 + (-2j)*(1-1j)


        Define a space with custom inner product:

        >>> weights = np.array([1., 2.])
        >>> def weighted_inner(x, y):
        ...     return np.vdot(weights * y.data, x.data)

        >>> c3w = Cn(2, inner=weighted_inner)
        >>> x = c3w.element(x)  # elements must be cast (no copy)
        >>> y = c3w.element(y)
        >>> c3w.inner(x, y) == 1*(5+1j)*1 + 2*(-2j)*(1-1j)

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
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> y = c3.element([1, 2+1j, 3-1j])
        >>> out = c3.element()
        >>> c3.multiply(x, y, out)  # out is returned

        >>> out

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
        >>> r3 = Rn(3)
        >>> x = r3.element([3, 5, 6])
        >>> y = r3.element([1, 2, 2])
        >>> out = r3.element()
        >>> r3.divide(x, y, out)  # out is returned

        >>> out

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
        >>> from numpy.linalg import norm
        >>> def dist(x, y, ord):
        ...     return norm(x - y, ord)

        >>> from functools import partial
        >>> dist2 = partial(dist, ord=2)
        >>> c3 = Cn(3, dist=dist2)
        >>> c3_same = Cn(3, dist=dist2)
        >>> c3  == c3_same


        Different ``dist`` functions result in different spaces - the
        same applies for ``norm`` and ``inner``:

        >>> dist1 = partial(dist, ord=1)
        >>> c3_1 = Cn(3, dist=dist1)
        >>> c3_2 = Cn(3, dist=dist2)
        >>> c3_1 == c3_2


        Be careful with Lambdas - they result in non-identical function
        objects:

        >>> c3_lambda1 = Cn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> c3_lambda2 = Cn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> c3_lambda1 == c3_lambda2


        An `Fn` space with the same data type is considered
        equal:

        >>> c3 = Cn(3)
        >>> f3_cdouble = Fn(3, dtype='complex128')
        >>> c3 == f3_cdouble

        """
        if other is self:
            return True

        return (NumpyTensorSet.__eq__(self, other) and
                self.weighting == other.weighting)

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.is_real_space:
            constructor_name = 'rtensors'
        elif self.is_complex_space:
            constructor_name = 'ctensors'
        else:
            constructor_name = 'tensor_space'

        inner_str = '{}'.format(self.shape)

        if (constructor_name == 'tensor_space' or
                self.dtype != self.default_dtype(self.field)):
            inner_str += ', {}'.format(dtype_repr(self.dtype))

        # TODO: default order class attribute?
        if self.order != 'C':
            inner_str += ", order='F'"

        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str
        return '{}({})'.format(constructor_name, inner_str)

    @property
    def element_type(self):
        """Type of elements in this space: `NumpyTensor`."""
        return NumpyTensor


class NumpyTensor(TensorBase, NumpyGeneralTensor):

    """Representation of a `NumpyTensorSpace` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        if not isinstance(space, NumpyTensorSpace):
            raise TypeError('`space` must be a `NumpyTensorSpace` instance, '
                            'got {!r}'.format(space))

        TensorBase.__init__(self, space)
        NumpyGeneralTensor.__init__(self, space, data)

    @property
    def real(self):
        """Real part of ``self``.

        Returns
        -------
        real : `NumpyTensor`
            Real part this element as an element of a `NumpyTensorSpace`
            with real data type.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> x.real

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
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> a = Rn(3).element([0, 0, 0])
        >>> x.real = a
        >>> x


        Other array-like types and broadcasting:

        >>> x.real = 1.0
        >>> x

        >>> x.real = [0, 2, -1]
        >>> x

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
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> x.imag

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
        >>> x = Cn(3).element([5+1j, 3, 2-2j])
        >>> a = Rn(3).element([0, 0, 0])
        >>> x.imag = a; print(x)


        Other array-like types and broadcasting:

        >>> x.imag = 1.0; print(x)

        >>> x.imag = [0, 2, -1]; print(x)

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
        >>> x = Cn(3).element([5+1j, 3, 2-2j])
        >>> y = x.conj(); print(y)


        The out parameter allows you to avoid a copy

        >>> z = Cn(3).element()
        >>> z_out = x.conj(out=z); print(z)

        >>> z_out is z


        It can also be used for in-place conj

        >>> x_out = x.conj(out=x); print(x)

        >>> x_out is x

        """
        if out is None:
            return self.space.element(self.data.conj())
        else:
            self.data.conj(out.data)
            return out

    def __ipow__(self, other):
        """Return ``self **= other``."""
        try:
            if other == int(other):
                return super().__ipow__(other)
        except TypeError:
            pass

        np.power(self.data, other, out=self.data)
        return self


def _weighting(weight, exponent, dist_using_inner=False):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weight):
        weighting = NumpyTensorSpaceConstWeighting(
            weight, exponent, dist_using_inner=dist_using_inner)
    elif weight is None:
        weighting = NumpyTensorSpaceNoWeighting(
            exponent, dist_using_inner=dist_using_inner)
    else:  # last possibility: make an array
        arr = np.asarray(weight)
        weighting = NumpyTensorSpaceArrayWeighting(
            arr, exponent, dist_using_inner=dist_using_inner)
    return weighting


def weighted_inner(weight):
    """Weighted inner product on `Fn` spaces as free function.

    Parameters
    ----------
    weight : scalar or `array-like`
        Weight of the inner product. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector and a
        2-dimensional array as a weighting matrix.

    Returns
    -------
    inner : `callable`
        Inner product function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    NumpyTensorSpaceConstWeighting, FnVectorWeighting, FnMatrixWeighting
    """
    return _weighting(weight, exponent=2.0).inner


def weighted_norm(weight, exponent=2.0):
    """Weighted norm on `Fn` spaces as free function.

    Parameters
    ----------
    weight : scalar or `array-like`
        Weight of the norm. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector and a
        2-dimensional array as a weighting matrix.
    exponent : positive `float`
        Exponent of the norm. If ``weight`` is a sparse matrix, only
        1.0, 2.0 and ``inf`` are allowed.

    Returns
    -------
    norm : `callable`
        Norm function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    NumpyTensorSpaceConstWeighting, FnVectorWeighting, FnMatrixWeighting
    """
    return _weighting(weight, exponent=exponent).norm


def weighted_dist(weight, exponent=2.0, use_inner=False):
    """Weighted distance on `Fn` spaces as free function.

    Parameters
    ----------
    weight : scalar or `array-like`
        Weight of the distance. A scalar is interpreted as a
        constant weight, a 1-dim. array as a weighting vector and a
        2-dimensional array as a weighting matrix.
    exponent : positive `float`
        Exponent of the norm. If ``weight`` is a sparse matrix, only
        1.0, 2.0 and ``inf`` are allowed.
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
    NumpyTensorSpaceConstWeighting, FnVectorWeighting, FnMatrixWeighting
    """
    return _weighting(weight, exponent=exponent,
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
    return np.linalg.norm(x.data, ord=p)


def _pnorm_diagweight(x, p, w):
    """Diagonally weighted p-norm implementation."""
    # This is faster than first applying the weights and then summing with
    # BLAS dot or nrm2
    xp = np.abs(x.data)
    if np.isfinite(p):
        xp = np.power(xp, p, out=xp)
        xp *= w  # w is a plain NumPy array
        return np.sum(xp) ** (1 / p)
    else:
        xp *= w
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
    return dot(x2.data, x1.data)


# TODO: implement intermediate weighting schemes with arrays that are
# broadcast, i.e. between scalar and full-blown in dimensionality?
# Possible use case: outer product of `ndim` 1-dim. arrays


class NumpyTensorSpaceArrayWeighting(ArrayWeightingBase):

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

              :math:`\\langle A, B\\rangle_W :=
              \\langle W \cdot A, B\\rangle =
              \\langle w \cdot a, b\\rangle =
              b^{\mathrm{H}} (w \cdot a)`,

          where :math:`a, b, w` are the "flattened" counterparts of
          tensors :math:`A, B, W`, respectively, :math:`b^{\mathrm{H}}`
          stands for transposed complex conjugate and :math:`w \cdot a`
          for element-wise multiplication.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\\infty`, the weighted norm is

              :math:`\\lVert A\\rVert_{W, \\infty} :=
              \\lVert W \cdot A\\rVert_{\\infty} =
              \\lVert w \cdot a\\rVert_{\\infty}`,

          otherwise it is (using point-wise exponentiation)

              :math:`\\lVert A\\rVert_{W, p} :=
              \\lVert W^{1/p} \cdot A\\rVert_{p} =
              \\lVert w^{1/p} \cdot a\\rVert_{\\infty}`.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

              :math:`\\lVert A\\rVert_{W, p} \\not\\to
              \\lVert A\\rVert_{W, \\infty} \quad (p \\to \\infty)`

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
            inner = _inner_default(x1 * self.vector, x2)
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
            return float(_pnorm_diagweight(x, self.exponent, self.vector))


class NumpyTensorSpaceConstWeighting(ConstWeightingBase):

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

              :math:`\\langle a, b\\rangle_c :=
              c \, \\langle a, b\\rangle_c =
              c \, b^{\mathrm{H}} a`,

          where :math:`b^{\mathrm{H}}` standing for transposed complex
          conjugate.

        - For other exponents, only norm and dist are defined. In the
          case of exponent :math:`\\infty`, the weighted norm is defined
          as
              :math:`\\lVert a \\rVert_{c, \\infty} :=
              c\, \\lVert a \\rVert_{\\infty}`,

          otherwise it is

              :math:`\\lVert a \\rVert_{c, p} :=
              c^{1/p}\, \\lVert a \\rVert_{p}`.

        - Note that this definition does **not** fulfill the limit
          property in :math:`p`, i.e.

              :math:`\\lVert a\\rVert_{c, p} \\not\\to
              \\lVert a \\rVert_{c, \\infty} \quad (p \\to \\infty)`

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
        if self._dist_using_inner:
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


class NumpyTensorSpaceNoWeighting(NoWeightingBase,
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


class NumpyTensorSpaceCustomInner(CustomInnerBase):

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


class NumpyTensorSpaceCustomNorm(CustomNormBase):

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


class NumpyTensorSpaceCustomDist(CustomDistBase):

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


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
