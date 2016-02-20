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

"""CPU implementations of ``n``-dimensional Cartesian spaces.

This is a default implementation of :math:`A^n` for an arbitrary set
:math:`A` as well as the real and complex spaces :math:`R^n` and
:math:`C^n`. The data is represented by NumPy arrays.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
from future.utils import native
standard_library.install_aliases()
from builtins import super

# External module imports
import ctypes
from numbers import Integral
from functools import partial
from math import sqrt
import numpy as np
import scipy.linalg as linalg
from scipy.sparse.base import isspmatrix

# ODL imports
from odl.operator.operator import Operator
from odl.space.base_ntuples import (
    NtuplesBase, NtuplesBaseVector, FnBase, FnBaseVector, FnWeightingBase)
from odl.util.utility import (
    dtype_repr, is_real_dtype, is_real_floating_dtype,
    is_complex_floating_dtype)
from odl.util.ufuncs import NtuplesUFuncs


__all__ = ('Ntuples', 'NtuplesVector', 'Fn', 'FnVector', 'Cn', 'Rn',
           'MatVecOperator',
           'FnMatrixWeighting', 'FnVectorWeighting', 'FnConstWeighting',
           'weighted_dist', 'weighted_norm', 'weighted_inner')


_BLAS_DTYPES = (np.dtype('float32'), np.dtype('float64'),
                np.dtype('complex64'), np.dtype('complex128'))


class Ntuples(NtuplesBase):

    """The set of n-tuples of arbitrary type."""

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : `array-like`, optional
            Input to initialize the new element.

            If ``inp`` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).

            If ``inp`` is a `numpy.ndarray` of shape ``(size,)``
            and the same data type as this space, the array is wrapped,
            not copied.
            Other array-like objects are copied.

        Returns
        -------
        element : `NtuplesVector`
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
        [w, b, w]
        >>> x.space
        Ntuples(3, '<U1')

        Construction from data pointer:

        >>> int3 = Ntuples(3, dtype='int')
        >>> x = int3.element([1, 2, 3])
        >>> y = int3.element(data_ptr=x.data_ptr)
        >>> print(y)
        [1, 2, 3]
        >>> y[0] = 5
        >>> print(x)
        [5, 2, 3]
        """
        if inp is None:
            if data_ptr is None:
                arr = np.empty(self.size, dtype=self.dtype)
                return self.element_type(self, arr)
            else:
                ctype_array_def = ctypes.c_byte * (self.size *
                                                   self.dtype.itemsize)
                as_ctype_array = ctype_array_def.from_address(data_ptr)
                as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
                arr = as_numpy_array.view(dtype=self.dtype)
                return self.element_type(self, arr)
        else:
            if data_ptr is None:
                arr = np.array(inp, copy=False, dtype=self.dtype, ndmin=1)
                if arr.shape != (self.size,):
                    raise ValueError('input shape {} not broadcastable to '
                                     'shape ({},).'.format(arr.shape,
                                                           self.size))

                return self.element_type(self, arr)
            else:
                raise ValueError('Cannot provide both `inp` and `data_ptr`')

    def zero(self):
        """Create a vector of zeros.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.zero()
        >>> x
        Cn(3).element([0j, 0j, 0j])
        """
        return self.element(np.zeros(self.size, dtype=self.dtype))

    def one(self):
        """Create a vector of ones.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.one()
        >>> x
        Cn(3).element([(1+0j), (1+0j), (1+0j)])
        """
        return self.element(np.ones(self.size, dtype=self.dtype))

    @property
    def element_type(self):
        """ `NtuplesVector` """
        return NtuplesVector


class NtuplesVector(NtuplesBaseVector):

    """Representation of an `Ntuples` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        if not isinstance(space, Ntuples):
            raise TypeError('{!r} not an `Ntuples` instance.'
                            ''.format(space))

        if not isinstance(data, np.ndarray):
            raise TypeError('data {!r} not a `numpy.ndarray` instance.'
                            ''.format(data))

        if data.dtype != space.dtype:
            raise TypeError('data {!r} not of dtype {!r}.'
                            ''.format(data, space.dtype))
        self._data = data

        NtuplesBaseVector.__init__(self, space)

    @property
    def data(self):
        """The raw `numpy.ndarray` representing the data."""
        return self._data

    def asarray(self, start=None, stop=None, step=None, out=None):
        """Extract the data of this array as a numpy array.

        Parameters
        ----------
        start : `int`, optional
            Start position. None means the first element.
        start : `int`, optional
            One element past the last element to be extracted.
            None means the last element.
        start : `int`, optional
            Step length. None means 1.
        out : `numpy.ndarray`, optional
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.

        Returns
        -------
        asarray : `numpy.ndarray`
            Numpy array of the same type as the space.

        Examples
        --------
        >>> import ctypes
        >>> vec = Ntuples(3, 'float').element([1, 2, 3])
        >>> vec.asarray()
        array([ 1.,  2.,  3.])
        >>> vec.asarray(start=1, stop=3)
        array([ 2.,  3.])

        Using the out parameter

        >>> out = np.empty((3,), dtype='float')
        >>> result = vec.asarray(out=out)
        >>> out
        array([ 1.,  2.,  3.])
        >>> result is out
        True
        """
        if out is None:
            return self.data[start:stop:step]
        else:
            out[:] = self.data[start:stop:step]
            return out

    @property
    def data_ptr(self):
        """A raw pointer to the data container.

        Examples
        --------
        >>> import ctypes
        >>> vec = Ntuples(3, 'int32').element([1, 2, 3])
        >>> arr_type = ctypes.c_int32 * 3
        >>> buffer = arr_type.from_address(vec.data_ptr)
        >>> arr = np.frombuffer(buffer, dtype='int32')
        >>> print(arr)
        [1 2 3]

        In-place modification via pointer:

        >>> arr[0] = 5
        >>> print(vec)
        [5, 2, 3]
        """
        return self._data.ctypes.data

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if all entries of other are equal to this
            vector's entries, `False` otherwise.

        Notes
        -----
        Space membership is not checked, hence vectors from
        different spaces can be equal.

        Examples
        --------
        >>> vec1 = Ntuples(3, int).element([1, 2, 3])
        >>> vec2 = Ntuples(3, int).element([-1, 2, 0])
        >>> vec1 == vec2
        False
        >>> vec2 = Ntuples(3, int).element([1, 2, 3])
        >>> vec1 == vec2
        True

        Space membership matters:

        >>> vec2 = Ntuples(3, float).element([1, 2, 3])
        >>> vec1 == vec2 or vec2 == vec1
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
        copy : `NtuplesVector`
            The deep copy

        Examples
        --------
        >>> vec1 = Ntuples(3, 'int').element([1, 2, 3])
        >>> vec2 = vec1.copy()
        >>> vec2
        Ntuples(3, 'int').element([1, 2, 3])
        >>> vec1 == vec2
        True
        >>> vec1 is vec2
        False
        """
        return self.space.element(self.data.copy())

    def __getitem__(self, indices):
        """Access values of this vector.

        Parameters
        ----------
        indices : `int` or `slice`
            The position(s) that should be accessed

        Returns
        -------
        values : scalar or `NtuplesVector`
            The value(s) at the index (indices)

        Examples
        --------
        >>> str_3 = Ntuples(3, dtype='U6')  # 6-char unicode
        >>> x = str_3.element(['a', 'Hello!', '0'])
        >>> print(x[0])
        a
        >>> print(x[1:3])
        [Hello!, 0]
        >>> x[1:3].space
        Ntuples(2, '<U6')
        """
        if isinstance(indices, Integral):
            return self.data[indices]  # single index
        else:
            arr = self.data[indices]
            return type(self.space)(len(arr), dtype=self.dtype).element(arr)

    def __setitem__(self, indices, values):
        """Return ``self[indices] = values``.

        Parameters
        ----------
        indices : `int` or `slice`
            The position(s) that should be set
        values : scalar, `array-like` or `NtuplesVector`
            The value(s) that are to be assigned.

            If ``indices`` is an integer, ``value`` must be scalar.

            If ``indices`` is a slice, ``value`` must be
            broadcastable to the size of the slice (same size,
            shape ``(1,)`` or single value).

        Returns
        -------
        `None`

        Examples
        --------
        >>> int_3 = Ntuples(3, 'int')
        >>> x = int_3.element([1, 2, 3])
        >>> x[0] = 5
        >>> x
        Ntuples(3, 'int').element([5, 2, 3])

        Assignment from array-like structures or another
        vector:

        >>> y = Ntuples(2, 'short').element([-1, 2])
        >>> x[:2] = y
        >>> x
        Ntuples(3, 'int').element([-1, 2, 3])
        >>> x[1:3] = [7, 8]
        >>> x
        Ntuples(3, 'int').element([-1, 7, 8])
        >>> x[:] = np.array([0, 0, 0])
        >>> x
        Ntuples(3, 'int').element([0, 0, 0])

        Broadcasting is also supported:

        >>> x[1:3] = -2.
        >>> x
        Ntuples(3, 'int').element([0, -2, -2])

        Array views are preserved:

        >>> y = x[::2]  # view into x
        >>> y[:] = -9
        >>> print(y)
        [-9, -9]
        >>> print(x)
        [-9, -2, -9]

        Be aware of unsafe casts and over-/underflows, there
        will be warnings at maximum.

        >>> x = Ntuples(2, 'int8').element([0, 0])
        >>> maxval = 255  # maximum signed 8-bit unsigned int
        >>> x[0] = maxval + 1
        >>> x
        Ntuples(2, 'int8').element([0, 0])
        """
        if ((values is self or values is self.data) and
                indices in (slice(None), Ellipsis)):
            return self
        if isinstance(values, NtuplesVector):
            return self.data.__setitem__(indices, values.data)
        else:
            return self.data.__setitem__(indices, values)

    @property
    def ufunc(self):
        """`NtuplesUFuncs`, access to numpy style ufuncs.

        Examples
        --------
        >>> r2 = Rn(2)
        >>> x = r2.element([1, -2])
        >>> x.ufunc.absolute()
        Rn(2).element([1.0, 2.0])

        These functions can also be used with broadcasting

        >>> x.ufunc.add(3)
        Rn(2).element([4.0, 1.0])

        and non-space elements

        >>> x.ufunc.subtract([3, 3])
        Rn(2).element([-2.0, -5.0])

        There is also support for various reductions (sum, prod, min, max)

        >>> x.ufunc.sum()
        -1.0

        They also support an out parameter

        >>> y = r2.element([3, 4])
        >>> out = r2.element()
        >>> result = x.ufunc.add(y, out=out)
        >>> result
        Rn(2).element([4.0, 2.0])
        >>> result is out
        True

        Notes
        -----
        These are optimized for use with ntuples and incur no overhead.
        """
        return NtuplesUFuncs(self)


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    `float` or `complex` data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.

    Parameters
    ----------
    x1,...,xN : `NtuplesBaseVector`
        The vectors to be tested for BLAS conformity
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


def _repr_space_funcs(space):
    """Return the string in the parentheses of repr for space funcs."""
    inner_str = ''

    weight = 1.0
    if space._space_funcs._dist_using_inner:
        inner_str += ', dist_using_inner=True'
    if isinstance(space._space_funcs, FnCustomInnerProduct):
        inner_str += ', inner=<custom inner>'
    elif isinstance(space._space_funcs, FnCustomNorm):
        inner_str += ', norm=<custom norm>'
    elif isinstance(space._space_funcs, FnCustomDist):
        inner_str += ', norm=<custom dist>'
    elif isinstance(space._space_funcs, FnConstWeighting):
        weight = space._space_funcs.const
        if weight != 1.0:
            inner_str += ', weight={}'.format(weight)
    elif isinstance(space._space_funcs, FnMatrixWeighting):
        weight = space._space_funcs.matrix
        inner_str += ', weight={!r}'.format(weight)

    exponent = space._space_funcs.exponent
    if exponent != 2.0:
        inner_str += ', exponent={}'.format(exponent)

    return inner_str


class Fn(FnBase, Ntuples):

    """The vector space F^n with vector multiplication.

    This space implements n-tuples of elements from a `Field` ``F``,
    which is usually the real or complex numbers.

    Its elements are represented as instances of the `FnVector` class.
    """

    def __init__(self, size, dtype, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive `int`
            The number of dimensions of the space
        dtype : `object`
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as `numpy.dtype` or as `string`.

            Only scalar data types are allowed.

        weight : `array-like` or `float`, optional
            Use weighted inner product, norm, and dist.

            float: Weighting by a constant

            array-like: Weighting by a matrix (2-dim. array) or a vector
            (1-dim. array, corresponds to a diagonal matrix). A matrix
            can also be given as a sparse matrix
            ( ``scipy.sparse.spmatrix``).

            Default: no weighting

            This option cannot be combined with ``dist``,
            ``norm`` or ``inner``.

        exponent : positive `float`, optional
            Exponent of the norm. For values other than 2.0, no
            inner product is defined.
            If ``weight`` is a sparse matrix, only 1.0, 2.0 and
            ``inf`` are allowed.

            This option is ignored if ``dist``, ``norm`` or
            ``inner`` is given.

            Default: 2.0

        dist : `callable`, optional
            The distance function defining a metric on
            :math:`\mathbb{F}^n`.
            It must accept two `FnVector` arguments and
            fulfill the following mathematical conditions for any
            three vectors :math:`x, y, z`:

            - :math:`d(x, y) = d(y, x)`
            - :math:`d(x, y) \geq 0`
            - :math:`d(x, y) = 0 \Leftrightarrow x = y`
            - :math:`d(x, y) \geq d(x, z) + d(z, y)`

            By default, ``dist(x, y)`` is calculated as
            ``norm(x - y)``. This creates an intermediate array
            ``x-y``, which can be
            avoided by choosing ``dist_using_inner=True``.

            This option cannot be combined with ``weight``,
            ``norm`` or ``inner``.

        norm : `callable`, optional
            The norm implementation. It must accept an
            `FnVector` argument, return a
            `float` and satisfy the following
            conditions for all vectors :math:`x, y` and scalars
            :math:`s`:

            - :math:`\lVert x\\rVert \geq 0`
            - :math:`\lVert x\\rVert = 0 \Leftrightarrow x = 0`
            - :math:`\lVert s x\\rVert = \lvert s \\rvert
              \lVert x\\rVert`
            - :math:`\lVert x + y\\rVert \leq \lVert x\\rVert +
              \lVert y\\rVert`.

            By default, ``norm(x)`` is calculated as
            ``inner(x, x)``.

            This option cannot be combined with ``weight``,
            ``dist`` or ``inner``.

        inner : `callable`, optional
            The inner product implementation. It must accept two
            `FnVector` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all vectors
            :math:`x, y, z` and scalars :math:`s`:

            - :math:`\langle x,y\\rangle =
              \overline{\langle y,x\\rangle}`
            - :math:`\langle sx, y\\rangle = s \langle x, y\\rangle`
            - :math:`\langle x+z, y\\rangle = \langle x,y\\rangle +
              \langle z,y\\rangle`
            - :math:`\langle x,x\\rangle = 0 \Leftrightarrow x = 0`

            This option cannot be combined with ``weight``,
            ``dist`` or ``norm``.

        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

            :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
            \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) :math:`x` and
            :math:`y`.

            This option can only be used if ``exponent`` is 2.0.

            Default: `False`.
        """
        Ntuples.__init__(self, size, dtype)
        FnBase.__init__(self, size, dtype)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)
        exponent = kwargs.pop('exponent', 2.0)
        dist_using_inner = bool(kwargs.pop('dist_using_inner', False))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weight)) < 3:
            raise ValueError('invalid combination of options `weight`, '
                             '`dist`, `norm` and `inner`.')
        if weight is not None:
            if np.isscalar(weight):
                self._space_funcs = FnConstWeighting(
                    weight, exponent, dist_using_inner=dist_using_inner)
            elif weight is None:
                pass
            elif isspmatrix(weight):
                self._space_funcs = FnMatrixWeighting(
                    weight, exponent, dist_using_inner=dist_using_inner)
            else:  # last possibility: make a matrix
                arr = np.asarray(weight)
                if arr.dtype == object:
                    raise ValueError('invalid weight argument {}.'
                                     ''.format(weight))
                if arr.ndim == 1:
                    self._space_funcs = FnVectorWeighting(
                        arr, exponent, dist_using_inner=dist_using_inner)
                elif arr.ndim == 2:
                    self._space_funcs = FnMatrixWeighting(
                        arr, exponent, dist_using_inner=dist_using_inner)
                else:
                    raise ValueError('array-like input {} is not 1- or '
                                     '2-dimensional.'.format(weight))

        elif dist is not None:
            self._space_funcs = FnCustomDist(dist)
        elif norm is not None:
            self._space_funcs = FnCustomNorm(norm)
        elif inner is not None:
            self._space_funcs = FnCustomInnerProduct(inner)
        else:  # all None -> no weighing
            self._space_funcs = FnNoWeighting(
                exponent, dist_using_inner=dist_using_inner)

    @property
    def exponent(self):
        """Exponent of the norm and distance."""
        return self._space_funcs.exponent

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of ``x1`` and ``x2``.

        Calculate ``out = a*x1 + b*x2`` using optimized BLAS
        routines if possible.

        Parameters
        ----------
        a, b : `FnBase.field`
            Scalars to multiply ``x1`` and ``x2`` with
        x1, x2 : `FnVector`
            Summands in the linear combination
        out : `FnVector`
            Vector to which the result is written

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
        Cn(3).element([(10-2j), (17-1j), (18.5+1.5j)])
        >>> out
        Cn(3).element([(10-2j), (17-1j), (18.5+1.5j)])
        """
        _lincomb(a, x1, b, x2, out, self.dtype)

    def _dist(self, x1, x2):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x1, x2 : `FnVector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
            Distance between the vectors

        Examples
        --------
        >>> from numpy.linalg import norm
        >>> c2_2 = Cn(2, dist=lambda x, y: norm(x - y, ord=2))
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)
        5.0

        >>> c2_2 = Cn(2, dist=lambda x, y: norm(x - y, ord=1))
        >>> x = c2_2.element([3+1j, 4])
        >>> y = c2_2.element([1j, 4-4j])
        >>> c2_2.dist(x, y)
        7.0
        """
        return self._space_funcs.dist(x1, x2)

    def _norm(self, x):
        """Calculate the norm of a vector.

        Parameters
        ----------
        x : `FnVector`
            The vector whose norm is calculated

        Returns
        -------
        norm : `float`
            Norm of the vector

        Examples
        --------
        >>> import numpy as np
        >>> c2_2 = Cn(2, norm=np.linalg.norm)  # 2-norm
        >>> x = c2_2.element([3+1j, 1-5j])
        >>> c2_2.norm(x)
        6.0

        >>> from functools import partial
        >>> c2_1 = Cn(2, norm=partial(np.linalg.norm, ord=1))
        >>> x = c2_1.element([3-4j, 12+5j])
        >>> c2_1.norm(x)
        18.0
        """
        return self._space_funcs.norm(x)

    def _inner(self, x1, x2):
        """Raw inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `FnVector`
            The vectors whose inner product is calculated

        Returns
        -------
        inner : `field` `element`
            Inner product of the vectors

        Examples
        --------
        >>> import numpy as np
        >>> c3 = Cn(2, inner=lambda x, y: np.vdot(y, x))
        >>> x = c3.element([5+1j, -2j])
        >>> y = c3.element([1, 1+1j])
        >>> c3.inner(x, y) == (5+1j)*1 + (-2j)*(1-1j)
        True

        Define a space with custom inner product:

        >>> weights = np.array([1., 2.])
        >>> def weighted_inner(x, y):
        ...     return np.vdot(weights * y.data, x.data)

        >>> c3w = Cn(2, inner=weighted_inner)
        >>> x = c3w.element(x)  # elements must be cast (no copy)
        >>> y = c3w.element(y)
        >>> c3w.inner(x, y) == 1*(5+1j)*1 + 2*(-2j)*(1-1j)
        True
        """
        return self._space_funcs.inner(x1, x2)

    def _multiply(self, x1, x2, out):
        """The entry-wise product of two vectors, assigned to out.

        Parameters
        ----------
        x1, x2 : `FnVector`
            Factors in the product
        out : `FnVector`
            Vector to which the result is written

        Returns
        -------
        `None`

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> y = c3.element([1, 2+1j, 3-1j])
        >>> out = c3.element()
        >>> c3.multiply(x, y, out)  # out is returned
        Cn(3).element([(5+1j), (6+3j), (4-8j)])
        >>> out
        Cn(3).element([(5+1j), (6+3j), (4-8j)])
        """
        np.multiply(x1.data, x2.data, out=out.data)

    def _divide(self, x1, x2, out):
        """The entry-wise division of two vectors, assigned to out.

        Parameters
        ----------
        x1, x2 : `FnVector`
            Dividend and divisor in the quotient
        out : `FnVector`
            Vector to which the result is written

        Returns
        -------
        `None`

        Examples
        --------
        >>> r3 = Rn(3)
        >>> x = r3.element([3, 5, 6])
        >>> y = r3.element([1, 2, 2])
        >>> out = r3.element()
        >>> r3.divide(x, y, out)  # out is returned
        Rn(3).element([3.0, 2.5, 3.0])
        >>> out
        Rn(3).element([3.0, 2.5, 3.0])
        """
        np.divide(x1.data, x2.data, out=out.data)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if other is an instance of this space's type
            with the same
            `NtuplesBase.size` and `NtuplesBase.dtype`,
            and identical distance function, otherwise `False`.

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
        True

        Different ``dist`` functions result in different spaces - the
        same applies for ``norm`` and ``inner``:

        >>> dist1 = partial(dist, ord=1)
        >>> c3_1 = Cn(3, dist=dist1)
        >>> c3_2 = Cn(3, dist=dist2)
        >>> c3_1 == c3_2
        False

        Be careful with Lambdas - they result in non-identical function
        objects:

        >>> c3_lambda1 = Cn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> c3_lambda2 = Cn(3, dist=lambda x, y: norm(x-y, ord=1))
        >>> c3_lambda1 == c3_lambda2
        False

        An `Fn` space with the same data type is considered
        equal:

        >>> c3 = Cn(3)
        >>> f3_cdouble = Fn(3, dtype='complex128')
        >>> c3 == f3_cdouble
        True
        """
        if other is self:
            return True

        return (Ntuples.__eq__(self, other) and
                self._space_funcs == other._space_funcs)

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        if self.is_rn:
            class_name = 'Rn'
            if self.dtype == np.float64:
                inner_str = '{}'.format(self.size)
            else:
                inner_str = '{}, {}'.format(self.size, self.dtype)
        elif self.is_cn:
            class_name = 'Cn'
            if self.dtype == np.complex128:
                inner_str = '{}'.format(self.size)
            else:
                inner_str = '{}, {}'.format(self.size, self.dtype)
        else:
            class_name = 'Fn'
            inner_str = '{}, {}'.format(self.size, dtype_repr(self.dtype))

        inner_str += _repr_space_funcs(self)
        return '{}({})'.format(class_name, inner_str)

    # Copy these to handle bug in ABCmeta
    zero = Ntuples.zero
    one = Ntuples.one

    @property
    def element_type(self):
        """Return `FnVector`."""
        return FnVector


class FnVector(FnBaseVector, NtuplesVector):

    """Representation of an `Fn` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        if not isinstance(space, Fn):
            raise TypeError('{!r} not an `Fn` instance.'
                            ''.format(space))

        FnBaseVector.__init__(self, space)
        NtuplesVector.__init__(self, space, data)

    @property
    def real(self):
        """The real part of this vector.

        Returns
        -------
        real : `FnVector` view with dtype
            The real part this vector as a vector in `Rn`

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> x.real
        Rn(3).element([5.0, 3.0, 2.0])

        The `Rn` vector is really a view, so changes affect
        the original array:

        >>> x.real *= 2
        >>> x
        Cn(3).element([(10+1j), (6+0j), (4-2j)])
        """
        rn = Rn(self.space.size, self.space.real_dtype)
        return rn.element(self.data.real)

    @real.setter
    def real(self, newreal):
        """The setter for the real part.

        This method is invoked by ``vec.real = other``.

        Parameters
        ----------
        newreal : `array-like` or scalar
            The new real part for this vector.

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> a = Rn(3).element([0, 0, 0])
        >>> x.real = a
        >>> x
        Cn(3).element([1j, 0j, -2j])

        Other array-like types and broadcasting:

        >>> x.real = 1.0
        >>> x
        Cn(3).element([(1+1j), (1+0j), (1-2j)])
        >>> x.real = [0, 2, -1]
        >>> x
        Cn(3).element([1j, (2+0j), (-1-2j)])
        """
        self.real.data[:] = newreal

    @property
    def imag(self):
        """The imaginary part of this vector.

        Returns
        -------
        imag : `FnVector`
            The imaginary part this vector as a vector in
            `Rn`

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> x.imag
        Rn(3).element([1.0, 0.0, -2.0])

        The `Rn` vector is really a view, so changes affect
        the original array:

        >>> x.imag *= 2
        >>> x
        Cn(3).element([(5+2j), (3+0j), (2-4j)])
        """
        rn = Rn(self.space.size, self.space.real_dtype)
        return rn.element(self.data.imag)

    @imag.setter
    def imag(self, newimag):
        """The setter for the imaginary part.

        This method is invoked by ``vec.imag = other``.

        Parameters
        ----------
        newreal : `array-like` or scalar
            The new imaginary part for this vector.

        Examples
        --------
        >>> x = Cn(3).element([5+1j, 3, 2-2j])
        >>> a = Rn(3).element([0, 0, 0])
        >>> x.imag = a; print(x)
        [(5+0j), (3+0j), (2+0j)]

        Other array-like types and broadcasting:

        >>> x.imag = 1.0; print(x)
        [(5+1j), (3+1j), (2+1j)]
        >>> x.imag = [0, 2, -1]; print(x)
        [(5+0j), (3+2j), (2-1j)]
        """
        self.imag.data[:] = newimag

    def conj(self, out=None):
        """The complex conjugate of this vector.

        Parameters
        ----------
        out : `FnVector`, optional
            Vector to which the complex conjugate is written.
            Must be an element of this vector's space.

        Returns
        -------
        out : `FnVector`
            The complex conjugate vector. If ``out`` was provided,
            the returned object is a reference to it.

        Examples
        --------
        >>> x = Cn(3).element([5+1j, 3, 2-2j])
        >>> y = x.conj(); print(y)
        [(5-1j), (3-0j), (2+2j)]

        The out parameter allows you to avoid a copy

        >>> z = Cn(3).element()
        >>> z_out = x.conj(out=z); print(z)
        [(5-1j), (3-0j), (2+2j)]
        >>> z_out is z
        True

        It can also be used for inplace conj
        >>> x_out = x.conj(out=x); print(x)
        [(5-1j), (3-0j), (2+2j)]
        >>> x_out is x
        True
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


def Cn(size, dtype='complex128', **kwargs):

    """The complex vector space :math:`C^n` with vector multiplication.

    Parameters
    ----------
    size : positive `int`
        The number of dimensions of the space
    dtype : `object`
        The data type of the storage array. Can be provided in any
        way the `numpy.dtype` function understands, most notably
        as built-in type, as one of NumPy's internal datatype
        objects or as string.

        Only complex floating-point data types are allowed.
    kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
        See `Fn`

    See also
    --------
    Fn : n-tuples over a field :math:`\mathbb{F}` with arbitrary scalar
        data type
    """

    cn = Fn(size, dtype, **kwargs)

    if not cn.is_cn:
        raise TypeError('data type {!r} not a complex floating-point type.'
                        ''.format(dtype))
    return cn


def Rn(size, dtype='float64', **kwargs):

    """The real vector space :math:`R^n` with vector multiplication.

     Parameters
    ----------
    size : positive `int`
        The number of dimensions of the space
    dtype : `object`
        The data type of the storage array. Can be provided in any
        way the `numpy.dtype` function understands, most notably
        as built-in type, as one of NumPy's internal datatype
        objects or as string.

        Only real floating-point data types are allowed.
    kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
        See `Fn`

    See also
    --------
    Fn : n-tuples over a field :math:`\mathbb{F}` with arbitrary scalar
        data type
    """

    rn = Fn(size, dtype, **kwargs)

    if not rn.is_rn:
        raise TypeError('data type {!r} not a real floating-point type.'
                        ''.format(dtype))
    return rn


class MatVecOperator(Operator):

    """Matrix multiply operator :math:`\mathbb{F}^n -> \mathbb{F}^m`."""

    def __init__(self, matrix, dom=None, ran=None):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : `array-like` or  ``scipy.sparse.spmatrix``
            Matrix representing the linear operator. Its shape must be
            ``(m, n)``, where ``n`` is the size of ``dom`` and ``m`` the
            size of ``ran``. Its dtype must be castable to the range
            ``dtype``.
        dom : `Fn`, optional
            Space on whose elements the matrix acts. If not provided,
            the domain is inferred from the matrix ``dtype`` and
            ``shape``. If provided, its dtype must be castable to the
            range dtype.
        ran : `Fn`, optional
            Space to which the matrix maps. If not provided,
            the domain is inferred from the matrix ``dtype`` and
            ``shape``.
        """
        if isspmatrix(matrix):
            self._matrix = matrix
        else:
            self._matrix = np.asarray(matrix)

        if self.matrix.ndim != 2:
            raise ValueError('matrix {} has {} axes instead of 2.'
                             ''.format(matrix, self.matrix.ndim))

        # Infer domain and range from matrix if necessary
        if is_real_floating_dtype(self.matrix):
            spc_type = Rn
        elif is_complex_floating_dtype(self.matrix):
            spc_type = Cn
        else:
            spc_type = Fn

        if dom is None:
            dom = spc_type(self.matrix.shape[1], dtype=self.matrix.dtype)
        elif not isinstance(dom, Fn):
            raise TypeError('domain {!r} is not an `Fn` instance.'
                            ''.format(dom))

        if ran is None:
            ran = spc_type(self.matrix.shape[0], dtype=self.matrix.dtype)
        elif not isinstance(ran, Fn):
            raise TypeError('range {!r} is not an `Fn` instance.'
                            ''.format(ran))

        # Check compatibility of matrix with domain and range
        if not np.can_cast(dom.dtype, ran.dtype):
            raise TypeError('domain data type {!r} cannot be safely cast to '
                            'range data type {!r}.'
                            ''.format(dom.dtype, ran.dtype))

        if self.matrix.shape != (ran.size, dom.size):
            raise ValueError('matrix shape {} does not match the required '
                             'shape {} of a matrix {} --> {}.'
                             ''.format(self.matrix.shape,
                                       (ran.size, dom.size),
                                       dom, ran))
        if not np.can_cast(self.matrix.dtype, ran.dtype):
            raise TypeError('matrix data type {!r} cannot be safely cast to '
                            'range data type {!r}.'
                            ''.format(matrix.dtype, ran.dtype))

        super().__init__(dom, ran, linear=True)

    @property
    def matrix(self):
        """Matrix representing this operator."""
        return self._matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isspmatrix(self.matrix)

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix."""
        if self.domain.field != self.range.field:
            raise NotImplementedError('adjoint not defined since fields '
                                      'of domain and range differ ({} != {}).'
                                      ''.format(self.domain.field,
                                                self.range.field))
        return MatVecOperator(self.matrix.conj().T,
                              dom=self.range, ran=self.domain)

    def _call(self, x, out=None):
        """Raw apply method on input, writing to given output."""
        if out is None:
            return self.range.element(self.matrix.dot(x.data))
        else:
            if self.matrix_issparse:
                # Unfortunately, there is no native in-place dot product for
                # sparse matrices
                out.data[:] = self.matrix.dot(x.data)
            else:
                self.matrix.dot(x.data, out=out.data)

    # TODO: repr and str


def _weighting(weight, exponent, dist_using_inner=False):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weight):
        weighting = FnConstWeighting(
            weight, exponent=exponent, dist_using_inner=dist_using_inner)
    elif isspmatrix(weight):
        weighting = FnMatrixWeighting(
            weight, exponent=exponent, dist_using_inner=dist_using_inner)
    else:
        weight_ = np.asarray(weight)
        if weight_.dtype == object:
            raise ValueError('bad weight {}'.format(weight))
        if weight_.ndim == 1:
            weighting = FnVectorWeighting(
                weight_, exponent=exponent, dist_using_inner=dist_using_inner)
        elif weight_.ndim == 2:
            weighting = FnMatrixWeighting(
                weight_, exponent=exponent, dist_using_inner=dist_using_inner)
        else:
            raise ValueError('array-like weight must have 1 or 2 dimensions, '
                             'but {} has {} dimensions.'
                             ''.format(weight, weight_.ndim))
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
    FnConstWeighting, FnVectorWeighting, FnMatrixWeighting
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
    FnConstWeighting, FnVectorWeighting, FnMatrixWeighting
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

        :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
        \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

        This avoids the creation of new arrays and is thus faster
        for large arrays. On the downside, it will not evaluate to
        exactly zero for equal (but not identical) :math:`x` and
        :math:`y`.

        Can only be used if ``exponent`` is 2.0.

    Returns
    -------
    dist : `callable`
        Distance function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    FnConstWeighting, FnVectorWeighting, FnMatrixWeighting
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


class FnMatrixWeighting(FnWeightingBase):

    """Matrix weighting for `Fn`.

    For exponent 2.0, a new weighted inner product with matrix :math:`W`
    is defined as

    :math:`\langle a, b\\rangle_W := b^H W a`

    with :math:`b^H` standing for transposed complex conjugate.

    For other exponents, only norm and dist are defined. In the case of
    exponent ``inf``, the weighted norm is

    :math:`\lVert a\\rVert_{W, \infty} := \lVert W a\\rVert_\infty`,

    otherwise it is

    :math:`\lVert a\\rVert_{W, p} := \lVert W^{1/p} a\\rVert_p`.

    Not that this definition does **not** fulfill the limit property
    in :math:`p`, i.e.

    :math:`\lim_{p\\to\\infty} \lVert a\\rVert_{W,p} =
    \lVert a\\rVert_\infty \\neq \lVert a\\rVert_{W,\infty}`

    unless :math:`W` is the identity matrix.

    The matrix must be Hermitian and posivive definite, otherwise it
    does not define an inner product or norm, respectively. This is not
    checked during initialization.
    """

    def __init__(self, matrix, exponent=2.0, dist_using_inner=False, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        matrix :  ``scipy.sparse.spmatrix`` or `array-like`, 2-dim.
            Square weighting matrix of the inner product
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
            If ``matrix`` is a sparse matrix, only 1.0, 2.0 and ``inf``
            are allowed.
        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

            :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
            \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) :math:`x` and
            :math:`y`.

            Can only be used if ``exponent`` is 2.0.
        precomp_mat_pow : `bool`
            If `True`, precompute the matrix power :math:`W^{1/p}`
            during initialization. This has no effect if
            ``exponent`` is 1.0, 2.0 or ``inf``.

            Default: `False`

        cache_mat_pow : `bool`
            If `True`, cache the matrix power :math:`W^{1/p}` during
            the first call to ``norm`` or ``dist``. This has no
            effect if ``exponent`` is 1.0, 2.0 or ``inf``.

            Default: `False`
        """
        precomp_mat_pow = kwargs.pop('precomp_mat_pow', False)
        cache_mat_pow = kwargs.pop('cache_mat_pow', True)
        FnWeightingBase.__init__(self, impl='numpy', exponent=exponent,
                                 dist_using_inner=dist_using_inner)

        if isspmatrix(matrix):
            self._matrix = matrix
        else:
            self._matrix = np.asarray(matrix)
            if self._matrix.dtype == object:
                raise ValueError('invalid matrix {}.'.format(matrix))
            elif self._matrix.ndim != 2:
                raise ValueError('matrix {} is {}-dimensional instead of '
                                 '2-dimensional.'
                                 ''.format(matrix, self._matrix.ndim))

        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError('matrix with shape {} not square.'
                             ''.format(self._matrix.shape))

        if (self.matrix_issparse and
                self.exponent not in (1.0, 2.0, float('inf'))):
            raise NotImplementedError('sparse matrices only supported for '
                                      'exponent 1.0, 2.0 or `inf`.')

        if self.exponent in (1.0, float('inf')):
            self._mat_pow = self.matrix
        elif precomp_mat_pow and self.exponent != 2.0:
            eigval, eigvec = linalg.eigh(self.matrix)
            eigval **= 1.0 / self.exponent
            self._mat_pow = (eigval * eigvec).dot(eigvec.conj().T)
        else:
            self._mat_pow = None

        self._cache_mat_pow = bool(cache_mat_pow)

    @property
    def matrix(self):
        """Weighting matrix of this inner product."""
        return self._matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isspmatrix(self.matrix)

    def matrix_isvalid(self):
        """Test if the matrix is positive definite Hermitian.

        This test tries to calculate a Cholesky decomposition and can
        be very time-consuming for large matrices. Sparse matrices are
        not supported.
        """
        if self.matrix_issparse:
            raise NotImplementedError('validation not supported for sparse '
                                      'matrices.')
        try:
            np.linalg.cholesky(self.matrix)
            return np.array_equal(self.matrix, self.matrix.conj().T)
        except np.linalg.LinAlgError:
            return False

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if other is an `FnMatrixWeighting` instance
            with **identical** matrix, `False` otherwise.

        See also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (isinstance(other, FnMatrixWeighting) and
                self.matrix is other.matrix and
                super().__eq__(other))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : `bool`
            `True` if other is an
            `FnWeightingBase` instance with the same `FnWeightingBase.impl`,
            which yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of this inner product's matrix with the matrix
            or constant of other.
        """
        # Optimization for equality
        if self == other:
            return True

        elif self.exponent != getattr(other, 'exponent', -1):
            return False

        elif isinstance(other, FnMatrixWeighting):
            if self.matrix.shape != other.matrix.shape:
                return False

            if self.matrix_issparse:
                if other.matrix_issparse:
                    # Optimization for different number of nonzero elements
                    if self.matrix.nnz != other.matrix.nnz:
                        return False
                    else:
                        # Most efficient out-of-the-box comparison
                        return (self.matrix != other.matrix).nnz == 0
                else:  # Worst case: compare against dense matrix
                    return np.array_equal(self.matrix.todense(), other.matrix)

            else:  # matrix of `self` is dense
                if other.matrix_issparse:
                    return np.array_equal(self.matrix, other.matrix.todense())
                else:
                    return np.array_equal(self.matrix, other.matrix)

        elif isinstance(other, FnVectorWeighting):
            if self.matrix_issparse:
                return (np.array_equiv(self.matrix.diagonal(),
                                       other.vector) and
                        np.array_equal(self.matrix.asformat('dia').offsets,
                                       np.array([0])))
            else:
                return np.array_equal(
                    self.matrix, other.vector * np.eye(self.matrix.shape[0]))

        elif isinstance(other, FnConstWeighting):
            if self.matrix_issparse:
                return (np.array_equiv(self.matrix.diagonal(), other.const) and
                        np.array_equal(self.matrix.asformat('dia').offsets,
                                       np.array([0])))
            else:
                return np.array_equal(
                    self.matrix, other.const * np.eye(self.matrix.shape[0]))
        else:
            return False

    def inner(self, x1, x2):
        """Calculate the matrix-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `FnVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : `float` or `complex`
            The inner product of the vectors
        """
        if self.exponent != 2.0:
            raise NotImplementedError('No inner product defined for '
                                      'exponent != 2 (got {}).'
                                      ''.format(self.exponent))
        else:
            inner = _inner_default(x1.space.element(self.matrix.dot(x1)), x2)
            if is_real_dtype(x1.dtype):
                return float(inner)
            else:
                return complex(inner)

    def norm(self, x):
        """Calculate the matrix-weighted norm of a vector.

        Parameters
        ----------
        x : `FnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : `float`
            The norm of the vector
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?
            if norm_squared < 0:
                norm_squared = 0.0  # Compensate for numerical error
            return sqrt(norm_squared)
        else:
            if self._mat_pow is None:
                # This case can only be reached if p != 1,2,inf
                if self.matrix_issparse:
                    raise NotImplementedError('sparse matrix powers not '
                                              'suppoerted.')
                else:
                    eigval, eigvec = linalg.eigh(self.matrix)
                    eigval **= 1.0 / self.exponent
                    mat_pow = (eigval * eigvec).dot(eigvec.conj().T)
                    if self._cache_mat_pow:
                        self._mat_pow = mat_pow
            else:
                mat_pow = self._mat_pow

            return float(_pnorm_default(x.space.element(mat_pow.dot(x)),
                                        self.exponent))

    def __repr__(self):
        """Return ``repr(self)``."""
        if self.matrix_issparse:
            inner_fstr = ('<{shape} sparse matrix, format {fmt!r}, {nnz} '
                          'stored entries>')
            fmt = self.matrix.format
            nnz = self.matrix.nnz
            if self.exponent != 2.0:
                inner_fstr += ', exponent={ex}'
            if self._dist_using_inner:
                inner_fstr += ', dist_using_inner=True'
        else:
            inner_fstr = '\n{matrix!r}'
            fmt = ''
            nnz = 0
            if self.exponent != 2.0:
                inner_fstr += ',\nexponent={ex}'
            if self._dist_using_inner:
                inner_fstr += ',\ndist_using_inner=True'
            else:
                inner_fstr += '\n'

        inner_str = inner_fstr.format(shape=self.matrix.shape, fmt=fmt,
                                      nnz=nnz, ex=self.exponent,
                                      matrix=self.matrix)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'Weighting: matrix =\n{}'.format(self.matrix)
        else:
            return 'Weighting: p = {}, matrix =\n{}'.format(self.exponent,
                                                            self.matrix)


class FnVectorWeighting(FnWeightingBase):

    """Vector weighting for `Fn`.

    For exponent 2.0, a new weighted inner product with vector :math:`w`
    is defined as

    :math:`\langle a, b\\rangle_w := b^H (w \odot a)`

    with :math:`b^H` standing for transposed complex conjugate and
    :math:`w \odot a` for element-wise multiplication.

    For other exponents, only norm and dist are defined. In the case of
    exponent ``inf``, the weighted norm is

    :math:`\lVert a\\rVert_{w,\infty}:=\lVert w\odot a\\rVert_\infty`,

    otherwise it is

    :math:`\lVert a\\rVert_{w, p} := \lVert w^{1/p}\odot a\\rVert_p`.

    Not that this definition does **not** fulfill the limit property
    in :math:`p`, i.e.

    :math:`\lim_{p\\to\\infty} \lVert a\\rVert_{w,p} =
    \lVert a\\rVert_\infty \\neq \lVert a\\rVert_{w,\infty}`

    unless :math:`w = (1,\dots,1)`.

    The vector may only have positive entries, otherwise it does not
    define an inner product or norm, respectively. This is not checked
    during initialization.
    """

    def __init__(self, vector, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `array-like`, one-dim.
            Weighting vector of the inner product
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
            If ``matrix`` is a sparse matrix, only 1.0, 2.0 and ``inf``
            are allowed.
        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

            :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
            \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) :math:`x` and
            :math:`y`.

            Can only be used if ``exponent`` is 2.0.
        """
        super().__init__(impl='numpy', exponent=exponent,
                         dist_using_inner=dist_using_inner)
        if not isinstance(vector, FnVector):
            self._vector = np.asarray(vector)
        else:
            self._vector = vector
        if self.vector.dtype == object:
            raise ValueError('invalid vector {}.'.format(vector))
        elif self.vector.ndim != 1:
            raise ValueError('vector {} is {}-dimensional instead of '
                             '1-dimensional.'
                             ''.format(vector, self._vector.ndim))

    @property
    def vector(self):
        """Weighting vector of this inner product."""
        return self._vector

    def vector_is_valid(self):
        """Test if the vector is a valid weight, i.e. positive."""
        return np.all(self.vector > 0)

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equals : `bool`
            `True` if other is an `FnVectorWeighting` instance
            with **identical** vector, `False` otherwise.

        See also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (isinstance(other, FnVectorWeighting) and
                self.vector is other.vector and
                super().__eq__(other))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : `bool`
            `True` if ``other`` is an
            `FnWeightingBase` instance with the same `FnWeightingBase.impl`,
            which yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of matrices/vectors/constant of this inner
            product and ``other``.
        """
        # Optimization for equality
        if self == other:
            return True
        elif (not isinstance(other, FnWeightingBase) or
              self.exponent != other.exponent):
            return False
        elif isinstance(other, FnMatrixWeighting):
            return other.equiv(self)
        elif isinstance(other, FnConstWeighting):
            return np.array_equiv(self.vector, other.const)
        else:
            return np.array_equal(self.vector, other.vector)

    def inner(self, x1, x2):
        """Calculate the vector weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `FnVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : `float` or `complex`
            The inner product of the two provided vectors
        """
        if self.exponent != 2.0:
            raise NotImplementedError('No inner product defined for '
                                      'exponent != 2 (got {}).'
                                      ''.format(self.exponent))
        else:
            inner = _inner_default(x1 * self.vector, x2)
            if is_real_dtype(x1.dtype):
                return float(inner)
            else:
                return complex(inner)

    def norm(self, x):
        """Calculate the vector-weighted norm of a vector.

        Parameters
        ----------
        x : `FnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : `float`
            The norm of the provided vector
        """
        if self.exponent == 2.0:
            norm_squared = self.inner(x, x).real  # TODO: optimize?!
            if norm_squared < 0:
                norm_squared = 0.0  # Compensate for numerical error
            return sqrt(norm_squared)
        else:
            return float(_pnorm_diagweight(x, self.exponent, self.vector))

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{vector!r}'
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(vector=self.vector, ex=self.exponent)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'Weighting: vector =\n{}'.format(self.vector)
        else:
            return 'Weighting: p = {}, vector =\n{}'.format(self.exponent,
                                                            self.vector)


class FnConstWeighting(FnWeightingBase):

    """Weighting of `Fn` by a constant.

    For exponent 2.0, a new weighted inner product with constant
    :math:`c` is defined as

    :math:`\langle a, b\\rangle_c := c\ b^H a`

    with :math:`b^H` standing for transposed complex conjugate.

    For other exponents, only norm and dist are defined. In the case of
    exponent ``inf``, the weighted norm is defined as

    :math:`\lVert a\\rVert_{c, \infty} := c \lVert a\\rVert_\infty`,

    otherwise it is

    :math:`\lVert a\\rVert_{c, p} := c^{1/p}  \lVert a\\rVert_p`.

    Not that this definition does **not** fulfill the limit property
    in :math:`p`, i.e.

    :math:`\lim_{p\\to\\infty} \lVert a\\rVert_{c,p} =
    \lVert a\\rVert_\infty \\neq \lVert a\\rVert_{c,\infty}`

    unless :math:`c = 1`.

    The constant :math:`c` must be positive.
    """

    def __init__(self, constant, exponent=2.0, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive `float`
            Weighting constant of the inner product.
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

            :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
            \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) :math:`x` and
            :math:`y`.

            Can only be used if ``exponent`` is 2.0.
        """
        FnWeightingBase.__init__(self, impl='numpy', exponent=exponent,
                                 dist_using_inner=dist_using_inner)
        self._const = float(constant)
        if self.const <= 0:
            raise ValueError('constant {} is not positive.'.format(constant))
        if not np.isfinite(self.const):
            raise ValueError('constant {} is invalid.'.format(constant))

    @property
    def const(self):
        """Weighting constant of this inner product."""
        return self._const

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : `bool`
            `True` if other is an `FnConstWeighting`
            instance with the same constant, `False` otherwise.
        """
        if other is self:
            return True

        return (isinstance(other, FnConstWeighting) and
                self.const == other.const and
                super().__eq__(other))

    def equiv(self, other):
        """Test if other is an equivalent weighting.

        Returns
        -------
        equivalent : `bool`
            `True` if other is an
            `FnWeightingBase` instance with the same `FnWeightingBase.impl`,
            which yields the same result as this inner product for any
            input, `False` otherwise. This is the same as equality
            if ``other`` is an `FnConstWeighting` instance,
            otherwise the `equiv` method of ``other`` is called.
        """
        if isinstance(other, FnConstWeighting):
            return self == other
        elif isinstance(other, FnWeightingBase):
            return other.equiv(self)
        else:
            return False

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `FnVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : `float` or `complex`
            The inner product of the two provided vectors
        """
        if self.exponent != 2.0:
            raise NotImplementedError('No inner product defined for '
                                      'exponent != 2 (got {}).'
                                      ''.format(self.exponent))
        else:
            inner = self.const * _inner_default(x1, x2)
            return x1.space.field.element(inner)

    def norm(self, x):
        """Calculate the constant-weighted norm of a vector.

        Parameters
        ----------
        x1 : `FnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : `float`
            The norm of the vector
        """
        if self.exponent == 2.0:
            return sqrt(self.const) * float(_norm_default(x))
        elif self.exponent == float('inf'):
            return self.const * float(_pnorm_default(x, self.exponent))
        else:
            return (self.const ** (1 / self.exponent) *
                    float(_pnorm_default(x, self.exponent)))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : `FnVector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
            The distance between the vectors
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

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{}'
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.const, ex=self.exponent)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'Weighting: const = {:.4}'.format(self.const)
        else:
            return 'Weighting: p = {}, const = {:.4}'.format(
                self.exponent, self.const)


class FnNoWeighting(FnConstWeighting):

    """Weighting of `Fn` with constant 1.

    For exponent 2.0, the unweighted inner product is defined as

    :math:`\langle a, b\\rangle := b^H a`

    with :math:`b^H` standing for transposed complex conjugate.

    For other exponents, only norm and dist are defined.
    """

    # Implement singleton pattern for efficiency in the default case
    _instance = None

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
            if not cls._instance:
                cls._instance = super().__new__(cls, *args, **kwargs)
            return cls._instance
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

            :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
            \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) :math:`x` and
            :math:`y`.

            Can only be used if ``exponent`` is 2.0.
        """
        if not hasattr(self, '_initialized'):
            FnConstWeighting.__init__(self, constant=1.0, exponent=exponent,
                                      dist_using_inner=dist_using_inner)
            self._initialized = True

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = ''
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'
        inner_str = inner_fstr.format(ex=self.exponent).lstrip(', ')

        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        if self.exponent == 2.0:
            return 'NoWeighting'
        else:
            return 'NoWeighting: p = {}'.format(self.exponent)


class FnCustomInnerProduct(FnWeightingBase):

    """Custom inner product on `Fn`."""

    def __init__(self, inner, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        inner : `callable`
            The inner product implementation. It must accept two
            `FnVector` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all vectors
            :math:`x, y, z` and scalars :math:`s`:

            - :math:`\langle x,y\\rangle =
              \overline{\langle y,x\\rangle}`
            - :math:`\langle sx, y\\rangle = s \langle x, y\\rangle`
            - :math:`\langle x+z, y\\rangle = \langle x,y\\rangle +
              \langle z,y\\rangle`
            - :math:`\langle x,x\\rangle = 0 \Leftrightarrow x = 0`

        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

            :math:`\lVert x-y \\rVert^2 = \lVert x \\rVert^2 +
            \lVert y \\rVert^2 - 2\Re \langle x, y \\rangle`.

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) :math:`x` and
            :math:`y`.
        """
        super().__init__(impl='numpy', exponent=2.0,
                         dist_using_inner=dist_using_inner)

        if not callable(inner):
            raise TypeError('inner product function {!r} not callable.'
                            ''.format(inner))
        self._inner_impl = inner

    @property
    def inner(self):
        """Custom inner product of this instance.."""
        return self._inner_impl

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : `bool`
            `True` if other is an `FnCustomInnerProduct`
            instance with the same inner product, `False` otherwise.
        """
        return (isinstance(other, FnCustomInnerProduct) and
                self.inner == other.inner and
                super().__eq__(other))

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{!r}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.inner)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        return self.__repr__()  # TODO: prettify?


class FnCustomNorm(FnWeightingBase):

    """Custom norm on `Fn`, removes ``inner``."""

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : `callable`
            The norm implementation. It must accept an
            `FnVector` argument, return a `float` and satisfy
            the following conditions for all vectors
            :math:`x, y` and scalars :math:`s`:

            - :math:`\lVert x\\rVert \geq 0`
            - :math:`\lVert x\\rVert = 0 \Leftrightarrow x = 0`
            - :math:`\lVert s x\\rVert = \lvert s \\rvert
              \lVert x\\rVert`
            - :math:`\lVert x + y\\rVert \leq \lVert x\\rVert +
              \lVert y\\rVert`.
        """
        super().__init__(impl='numpy', exponent=1.0, dist_using_inner=False)

        if not callable(norm):
            raise TypeError('norm function {!r} not callable.'
                            ''.format(norm))
        self._norm_impl = norm

    def inner(self, x1, x2):
        """Inner product is not defined for custom distance."""
        raise NotImplementedError

    @property
    def norm(self):
        """Custom norm of this instance.."""
        return self._norm_impl

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : `bool`
            `True` if other is an `FnCustomNorm`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, FnCustomNorm) and
                self.norm == other.norm and
                super().__eq__(other))

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.norm)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        return self.__repr__()  # TODO: prettify?


class FnCustomDist(FnWeightingBase):

    """Custom distance on `Fn`, removes ``norm`` and ``inner``."""

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : `callable`
            The distance function defining a metric on
            :math:`\mathbb{F}^n`.
            It must accept two `FnVector` arguments and
            fulfill the following mathematical conditions for any
            three vectors :math:`x, y, z`:

            - :math:`d(x, y) = d(y, x)`
            - :math:`d(x, y) \geq 0`
            - :math:`d(x, y) = 0 \Leftrightarrow x = y`
            - :math:`d(x, y) \geq d(x, z) + d(z, y)`
        """
        super().__init__(impl='numpy', exponent=1.0, dist_using_inner=False)

        if not callable(dist):
            raise TypeError('distance function {!r} not callable.'
                            ''.format(dist))
        self._dist_impl = dist

    @property
    def dist(self):
        """Custom distance of this instance.."""
        return self._dist_impl

    def inner(self, x1, x2):
        """Inner product is not defined for custom distance."""
        raise NotImplementedError

    def norm(self, x):
        """Norm is not defined for custom distance."""
        raise NotImplementedError

    def __eq__(self, other):
        """Return ``self == other``.

        Returns
        -------
        equal : `bool`
            `True` if other is an `FnCustomDist`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, FnCustomDist) and
                self.dist == other.dist and
                super().__eq__(other))

    def __repr__(self):
        """Return ``repr(self)``."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.dist)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """Return ``str(self)``."""
        return self.__repr__()  # TODO: prettify?


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
