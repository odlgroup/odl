# Copyright 2014, 2015 The ODL development group
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

"""CPU implementations of `n`-dimensional Cartesian spaces.

This is a default implementation of :math:`A^n` for an arbitrary set
:math:`A` as well as the real and complex spaces :math:`R^n` and
:math:`C^n`. The latter two each come in a basic version with vector
multiplication only and as metric, normed, Hilbert and Euclidean space
variants. The data is represented by NumPy arrays.

List of classes
---------------

+-------------+--------------+----------------------------------------+
|Class name   |Direct        |Description                             |
|             |Ancestors     |                                        |
+=============+==============+========================================+
|`Ntuples`    |`Set`         |Basic class of `n`-tuples where each    |
|             |              |entry is of the same type               |
+-------------+--------------+----------------------------------------+
|`Fn`         |`EuclideanCn` |`HilbertRn` with the standard inner     |
|             |              |(dot) product                           |
+-------------+--------------+----------------------------------------+
|`Cn`         |`Ntuples`,    |`n`-tuples of complex numbers with      |
|             |`Algebra`     |vector-vector multiplication            |
+-------------+--------------+----------------------------------------+
|`Rn`         |`Cn`          |`n`-tuples of real numbers with         |
|             |              |vector-vector multiplication            |
+-------------+--------------+----------------------------------------+
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import int, super
from future.utils import with_metaclass

# External module imports
from abc import ABCMeta
from math import sqrt
import numpy as np
import scipy as sp
import ctypes
from scipy.linalg.blas import get_blas_funcs
import platform

# ODL imports
from odl.operator.operator import Operator
from odl.space.base_ntuples import NtuplesBase, FnBase, _FnWeightingBase
from odl.util.utility import dtype_repr
from odl.util.utility import is_real_dtype, is_complex_dtype


__all__ = ('Ntuples', 'Fn', 'Cn', 'Rn', 'MatVecOperator')


_TYPE_MAP_C2R = {np.dtype('float32'): np.dtype('float32'),
                 np.dtype('float64'): np.dtype('float64'),
                 np.dtype('complex64'): np.dtype('float32'),
                 np.dtype('complex128'): np.dtype('float64')}

_TYPE_MAP_R2C = {np.dtype('float32'): np.dtype('complex64'),
                 np.dtype('float64'): np.dtype('complex128')}

if platform.system() == 'Linux':
    _TYPE_MAP_C2R.update({np.dtype('float128'): np.dtype('float128'),
                          np.dtype('complex256'): np.dtype('float128')})
    _TYPE_MAP_R2C.update({np.dtype('float128'): np.dtype('complex256')})


_BLAS_DTYPES = (np.dtype('float32'), np.dtype('float64'),
                np.dtype('complex64'), np.dtype('complex128'))


class Ntuples(NtuplesBase):

    """The set of `n`-tuples of arbitrary type.

    See also
    --------
    See the module documentation for attributes, methods etc.
    """

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : array-like or scalar, optional
            Input to initialize the new element.

            If `inp` is `None`, an empty element is created with no
            guarantee of its state (memory allocation only).

            If `inp` is a `numpy.ndarray` of shape `(size,)` and the
            same data type as this space, the array is wrapped, not
            copied.
            Other array-like objects are copied (with broadcasting
            if necessary).

            If a single value is given, it is copied to all entries.

        Returns
        -------
        element : `Ntuples.Vector`
            The new element created (from `inp`).

        Note
        ----
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
                return self.Vector(self, arr)
            else:
                ctype_array_def = ctypes.c_byte * (self.size *
                                                   self.dtype.itemsize)
                as_ctype_array = ctype_array_def.from_address(data_ptr)
                as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
                arr = as_numpy_array.view(dtype=self.dtype)
                return self.Vector(self, arr)
        else:
            if data_ptr is None:
                if isinstance(inp, Ntuples.Vector):
                    inp = inp.data.astype(self.dtype, copy=True)
                else:
                    inp = np.atleast_1d(inp).astype(self.dtype, copy=False)

                if inp.shape == (1,):
                    arr = np.empty(self.size, dtype=self.dtype)
                    arr[:] = inp
                elif inp.shape == (self.size,):
                    arr = inp
                else:
                    raise ValueError('input shape {} not broadcastable to '
                                     'shape ({},).'.format(inp.shape,
                                                           self.size))

                return self.Vector(self, arr)
            else:
                raise ValueError('Cannot provide both `inp` and `data_ptr`')

    class Vector(NtuplesBase.Vector):

        """Representation of an `Ntuples` element.

        See also
        --------
        See the module documentation for attributes, methods etc.
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, Ntuples):
                raise TypeError('{!r} not an `Ntuples` instance.'
                                ''.format(space))

            if not isinstance(data, np.ndarray):
                raise TypeError('data {!r} not a `numpy.ndarray` instance.'
                                ''.format(data))

            if data.dtype != space.dtype:
                raise TypeError('data {!r} not of dtype `{!r}`.'
                                ''.format(data, space.dtype))

            self._data = data

            super().__init__(space)

        @property
        def data(self):
            """The raw numpy array representing the data."""
            return self._data

        def asarray(self, start=None, stop=None, step=None, out=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            start : int, optional (default: `None`)
                Start position. None means the first element.
            start : int, optional (default: `None`)
                One element past the last element to be extracted.
                None means the last element.
            start : int, optional (default: `None`)
                Step length. None means 1.
            out : `ndarray`, optional (default: `None`)
                Array in which the result should be written in-place.
                Has to be contiguous and of the correct dtype.

            Returns
            -------
            asarray : `ndarray`
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
            """`vec.__eq__(other) <==> vec == other`.

            Returns
            -------
            equals :  bool
                `True` if all entries of `other` are equal to this
                vector's entries, `False` otherwise.

            Note
            ----
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

            Returns
            -------
            copy : `Ntuples.Vector`
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
            indices : int or slice
                The position(s) that should be accessed

            Returns
            -------
            values : `self.dtype.element` or `space.Vector`
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
            try:
                return self.data[int(indices)]  # single index
            except TypeError:
                arr = self.data[indices]
                return type(self.space)(len(arr),
                                        dtype=self.dtype).element(arr)

        def __setitem__(self, indices, values):
            """Set values of this vector.

            Parameters
            ----------
            indices : int or slice
                The position(s) that should be set
            values : {scalar, array-like, `Ntuples.Vector`}
                The value(s) that are to be assigned.

                If `indices` is an integer, `value` must be single value.

                If `indices` is a slice, `value` must be
                broadcastable to the size of the slice (same size,
                shape (1,) or single value).

            Returns
            -------
            None

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
            if isinstance(values, Ntuples.Vector):
                return self.data.__setitem__(indices, values.data)
            else:
                return self.data.__setitem__(indices, values)


def _blas_is_applicable(*args):
    """Whether BLAS routines can be applied or not.

    BLAS routines are available for single and double precision
    float or complex data only. If the arrays are non-contiguous,
    BLAS methods are usually slower, and array-writing routines do
    not work at all. Hence, only contiguous arrays are allowed.

    Parameters
    ----------
    x1,...,xN : `NtuplesBase.Vector`
        The vectors to be tested for BLAS conformity
    """
    if len(args) == 0:
        return False

    return (all(x.dtype == args[0].dtype for x in args) and
            all(x.dtype in _BLAS_DTYPES for x in args) and
            all(x.data.flags.contiguous for x in args))


def _lincomb(a, x1, b, x2, out, dtype):
    """Raw linear combination depending on data type."""
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
        # pylint: disable=unbalanced-tuple-unpacking
        axpy, scal, copy = get_blas_funcs(
            ['axpy', 'scal', 'copy'], arrays=(x1.data, x2.data))
    else:
        axpy, scal, copy = (fallback_axpy, fallback_scal, fallback_copy)

    if x1 is x2 and b != 0:
        # x1 is aligned with x2 -> out = (a+b)*x1
        _lincomb(a+b, x1, 0, x1, out, dtype)
    elif out is x1 and out is x2:
        # All the vectors are aligned -> out = (a+b)*out
        scal(a+b, out.data, len(out))
    elif out is x1:
        # out is aligned with x1 -> out = a*out + b*x2
        if a != 1:
            scal(a, out.data, len(out))
        if b != 0:
            axpy(x2.data, out.data, len(out), b)
    elif out is x2:
        # out is aligned with x2 -> out = a*x1 + b*out
        if b != 1:
            scal(b, out.data, len(out))
        if a != 0:
            axpy(x1.data, out.data, len(out), a)
    else:
        # We have exhausted all alignment options, so x1 != x2 != out
        # We now optimize for various values of a and b
        if b == 0:
            if a == 0:  # Zero assignment -> out = 0
                out.data[:] = 0
            else:  # Scaled copy -> out = a*x1
                copy(x1.data, out.data, len(out))
                if a != 1:
                    scal(a, out.data, len(out))
        else:
            if a == 0:  # Scaled copy -> out = b*x2
                copy(x2.data, out.data, len(out))
                if b != 1:
                    scal(b, out.data, len(out))

            elif a == 1:  # No scaling in x1 -> out = x1 + b*x2
                copy(x1.data, out.data, len(out))
                axpy(x2.data, out.data, len(out), b)
            else:  # Generic case -> out = a*x1 + b*x2
                copy(x2.data, out.data, len(out))
                if b != 1:
                    scal(b, out.data, len(out))
                axpy(x1.data, out.data, len(out), a)


def _repr_space_funcs(space):
    inner_str = ''

    weight = 1.0
    if space._space_funcs._dist_using_inner:
        inner_str += ', dist_using_inner=True'
    if isinstance(space._space_funcs, _FnCustomInnerProduct):
        inner_str += ', inner=<custom inner>'
    elif isinstance(space._space_funcs, _FnCustomNorm):
        inner_str += ', norm=<custom norm>'
    elif isinstance(space._space_funcs, _FnCustomDist):
        inner_str += ', norm=<custom dist>'
    elif isinstance(space._space_funcs, _FnConstWeighting):
        weight = space._space_funcs.const
        if weight != 1.0:
            inner_str += ', weight={}'.format(weight)
    elif isinstance(space._space_funcs, _FnMatrixWeighting):
        weight = space._space_funcs.matrix
        inner_str += ', weight={!r}'.format(weight)

    return inner_str


class Fn(FnBase, Ntuples):

    """The vector space :math:`F^n` with vector multiplication.

    This space implements n-tuples of elements from a field :math:`F`,
    which can be the real or the complex numbers.

    Its elements are represented as instances of the inner `Fn.Vector`
    class.

    See also
    --------
    See the module documentation for attributes, methods etc.
    """

    def __init__(self, size, dtype, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only scalar data types are allowed.

        kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
            'weight' : matrix, float or `None`
                Use weighted inner product, norm, and dist.

                `None` (default) : Use the standard unweighted functions

                float : Use functions weighted by a constant

                matrix : Use functions weighted by a matrix. The matrix
                can be dense (`numpy.matrix`) or sparse
                (`scipy.sparse.spmatrix`).

                This option cannot be combined with `dist`, `norm` or
                `inner`.

            'dist' : callable, optional
                The distance function defining a metric on :math:`F^n`.
                It must accept two `Fn.Vector` arguments and fulfill the
                following conditions for any vectors `x`, `y` and `z`:

                - `dist(x, y) == dist(y, x)`
                - `dist(x, y) >= 0`
                - `dist(x, y) == 0` (approx.) if and only if `x == y`
                  (approx.)
                - `dist(x, y) <= dist(x, z) + dist(z, y)`

                By default, `dist(x, y)` is calculated as `norm(x - y)`.
                This creates an intermediate array `x-y`, which can be
                avoided by choosing `dist_using_inner=True`.

                This option cannot be combined with `weight`, `norm`
                or `inner`.

            'norm' : callable, optional (Default: `sqrt(inner(x,y))`)
                The norm implementation. It must accept an `Fn.Vector`
                argument, return a `RealNumber` and satisfy the
                following properties:

                - `norm(x) >= 0`
                - `norm(x) == 0` (approx.) only if `x == 0` (approx.)
                - `norm(s * x) == abs(s) * norm(x)` for `s` scalar
                - `norm(x + y) <= norm(x) + norm(y)`

                This option cannot be combined with `weight`, `dist`
                or `inner`.

            'inner' : callable, optional
                The inner product implementation. It must accept two
                `Fn.Vector` arguments, return a complex number and
                satisfy the following conditions for all vectors `x`,
                `y` and `z` and scalars `s`:

                 - `inner(x, y) == conjugate(inner(y, x))`
                 - `inner(s * x, y) == s * inner(x, y)`
                 - `inner(x + z, y) == inner(x, y) + inner(z, y)`
                 - `inner(x, x) == 0` (approx.) only if `x == 0`
                   (approx.)

                This option cannot be combined with `weight`, `dist`
                or `norm`.

            'dist_using_inner' : bool, optional  (Default: `False`)
                Calculate `dist(x, y)` as

                `sqrt(norm(x)**2 + norm(y)**2 - 2 * inner(x, y).real)`

                This avoids the creation of new arrays and is thus
                faster for large arrays. On the downside, it will not
                evaluate to exactly zero for equal (but not identical)
                `x` and `y`.
        """
        super().__init__(size, dtype)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)
        dist_using_inner = bool(kwargs.pop('dist_using_inner', False))

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if (dist, norm, inner, weight).count(None) < 3:
            raise ValueError('invalid combination of options `weight`, '
                             '`dist`, `norm` and `inner`.')
        if weight is not None:
            if np.isscalar(weight):
                self._space_funcs = _FnConstWeighting(
                    weight, dist_using_inner=dist_using_inner)
            elif isinstance(weight, (np.matrix, sp.sparse.spmatrix)):
                self._space_funcs = _FnMatrixWeighting(
                    weight, dist_using_inner=dist_using_inner)
            elif weight is None:
                pass
            else:
                raise ValueError('invalid weight argument {!r}.'
                                 ''.format(weight))
        elif dist is not None:
            self._space_funcs = _FnCustomDist(dist)
        elif norm is not None:
            self._space_funcs = _FnCustomNorm(norm)
        elif inner is not None:
            self._space_funcs = _FnCustomInnerProduct(inner)
        else:  # all None -> no weighing
            self._space_funcs = _FnNoWeighting()

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of `x` and `y`.

        Calculate `out = a * x1 + b * x2` using optimized BLAS routines if
        possible.

        Parameters
        ----------
        a, b : `field` element
            Scalar to multiply `x` and `y` with.
        x1, x2 : `Fn.Vector`
            The summands
        out : `Fn.Vector`
            The Vector that the result is written to.

        Returns
        -------
        None

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([1+1j, 2-1j, 3])
        >>> y = c3.element([4+0j, 5, 6+0.5j])
        >>> out = c3.element()
        >>> c3.lincomb(2j, x, 3-1j, y, out)
        >>> out
        Cn(3).element([(10-2j), (17-1j), (18.5+1.5j)])
        """
        _lincomb(a, x1, b, x2, out, self.dtype)

    def _dist(self, x1, x2):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            The vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
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
        x : `Fn.Vector`
            The vector whose norm is calculated

        Returns
        -------
        norm : float
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

        x1, x2 : `Cn.Vector`
            The vectors whose inner product is calculated

        Returns
        -------
        inner : `complex`
            Inner product of `x1` and `x2`.

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
        """The entry-wise product of two vectors, assigned to `out`.

        Parameters
        ----------
        x1, x2 : `Cn.Vector`
            Factors in the product
        out : `Cn.Vector`
            The result vector

        Returns
        -------
        None

        Examples
        --------
        >>> c3 = Cn(3)
        >>> x = c3.element([5+1j, 3, 2-2j])
        >>> y = c3.element([1, 2+1j, 3-1j])
        >>> out = c3.element()
        >>> c3.multiply(x, y, out)
        >>> out
        Cn(3).element([(5+1j), (6+3j), (4-8j)])
        """
        if out is x1 and out is x2:  # out = out*out
            out.data[:] *= out.data
        elif out is x1:  # out = out*x2
            out.data[:] *= x2.data
        elif out is x2:  # out = out*x1
            out.data[:] *= x1.data
        else:  # out = x1*x2
            out.data[:] = x1.data
            out.data[:] *= x2.data

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

    def __eq__(self, other):
        """`s.__eq__(other) <==> s == other`.

        Returns
        -------
        equals : bool
            `True` if `other` is an instance of this space's type
            with the same `size` and `dtype`, and **identical**
            distance function, otherwise `False`.

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

        Different `dist` functions result in different spaces - the
        same applies for `norm` and `inner`:

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

        An `Fn` space with the same data type is considered equal:

        >>> c3 = Cn(3)
        >>> f3_cdouble = Fn(3, dtype='complex128')
        >>> c3 == f3_cdouble
        True
        """
        if other is self:
            return True

        return (super().__eq__(other) and
                self._space_funcs == other._space_funcs)

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_str = '{}, {}'.format(self.size, dtype_repr(self.dtype))
        inner_str += _repr_space_funcs(self)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    class Vector(FnBase.Vector, Ntuples.Vector):

        """Representation of an `Fn` element.

        See also
        --------
        See the module documentation for attributes, methods etc.
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, Fn):
                raise TypeError('{!r} not an `Fn` instance.'
                                ''.format(space))

            if not isinstance(data, np.ndarray):
                raise TypeError('data {!r} not a `numpy.ndarray` instance.'
                                ''.format(data))
            super().__init__(space, data)


class Cn(Fn):

    """The complex vector space :math:`C^n` with vector multiplication.

    Its elements are represented as instances of the inner `Cn.Vector`
    class.

    See also
    --------
    Fn : n-tuples over a field :math:`F` with arbitrary scalar data type
    """

    def __init__(self, size, dtype=np.complex128, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only complex floating-point data types are allowed.
        kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
            See `Fn`
        """
        super().__init__(size, dtype, **kwargs)

        if not is_complex_dtype(self._dtype):
            raise TypeError('data type {} not a complex floating-point type.'
                            ''.format(dtype))
        self._real_dtype = _TYPE_MAP_C2R[self._dtype]

    @property
    def real_dtype(self):
        """The corresponding real data type of this space."""
        return self._real_dtype

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_fstr = '{}'
        if self.dtype != np.complex128:
            inner_fstr += ', {dtype}'

        inner_str = inner_fstr.format(self.size, dtype=dtype_repr(self.dtype))
        inner_str += _repr_space_funcs(self)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`cn.__str__() <==> str(cn)`."""
        if self.dtype == np.complex128:
            return 'Cn({})'.format(self.size)
        else:
            return 'Cn({}, {})'.format(self.size, self.dtype)

    class Vector(Fn.Vector):
        """Representation of a `Cn` element.

        See also
        --------
        See the module documentation for attributes, methods etc.
        """

        @property
        def real(self):
            """The real part of this vector.

            Returns
            -------
            real : `Rn.Vector` view
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

            This method is invoked by `vec.real = other`.

            Parameters
            ----------
            newreal : array-like or scalar
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
            imag : `Rn.Vector`
                The imaginary part this vector as a vector in `Rn`

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

            This method is invoked by `vec.imag = other`.

            Parameters
            ----------
            newreal : array-like or scalar
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


class Rn(Fn):

    """The real vector space :math:`R^n` with vector multiplication.

    Its elements are represented as instances of the inner `Rn.Vector`
    class.

    See also
    --------
    Fn : n-tuples over a field :math:`F` with arbitrary scalar data type
    """

    def __init__(self, size, dtype=np.float64, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only real floating-point data types are allowed.
        kwargs : {'weight', 'dist', 'norm', 'inner', 'dist_using_inner'}
            See `Fn`
        """
        super().__init__(size, dtype, **kwargs)

        if not is_real_dtype(self._dtype):
            raise TypeError('data type {} not a real floating-point type.'
                            ''.format(dtype))

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_fstr = '{}'
        if self.dtype != np.float64:
            inner_fstr += ', {dtype}'

        inner_str = inner_fstr.format(self.size, dtype=dtype_repr(self.dtype))
        inner_str += _repr_space_funcs(self)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`rn.__str__() <==> str(rn)`."""
        if self.dtype == np.float64:
            return 'Rn({})'.format(self.size)
        else:
            return 'Rn({}, {})'.format(self.size, self.dtype)


class MatVecOperator(Operator):
    # TODO: move to some default operator place

    """Operator :math:`F^n -> F^m` represented by a matrix."""

    def __init__(self, dom, ran, matrix):
        """Initialize a new instance.

        Parameters
        ----------
        dom : `Fn`
            Space on whose elements the matrix acts. Its dtype must be
            castable to the range dtype.
        ran : `Fn`
            Space to which the matrix maps
        matrix : array-like or scipy.sparse.spmatrix
            Matrix representing the linear operator. Its shape must be
            `(m, n)`, where `n` is the size of `dom` and `m` the size
            of `ran`. Its dtype must be castable to the range dtype.
        """
        super().__init__(dom, ran, linear=True)
        if not isinstance(dom, Fn):
            raise TypeError('domain {!r} is not an `Fn` instance.'
                            ''.format(dom))
        if not isinstance(ran, Fn):
            raise TypeError('range {!r} is not an `Fn` instance.'
                            ''.format(ran))
        if not np.can_cast(dom.dtype, ran.dtype):
            raise TypeError('domain data type {} cannot be safely cast to '
                            'range data type {}.'
                            ''.format(dom.dtype, ran.dtype))

        if isinstance(matrix, sp.sparse.spmatrix):
            self._matrix = matrix
        else:
            self._matrix = np.asmatrix(matrix)

        if self._matrix.shape != (ran.size, dom.size):
            raise ValueError('matrix shape {} does not match the required '
                             'shape {} of a matrix {} --> {}.'
                             ''.format(self._matrix.shape,
                                       (ran.size, dom.size),
                                       dom, ran))
        if not np.can_cast(self._matrix.dtype, ran.dtype):
            raise TypeError('matrix data type {} cannot be safely cast to '
                            'range data type {}.'
                            ''.format(matrix.dtype, ran.dtype))

    @property
    def matrix(self):
        """Matrix representing this operator."""
        return self._matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isinstance(self.matrix, sp.sparse.spmatrix)

    @property
    def adjoint(self):
        """Adjoint operator represented by the adjoint matrix."""
        if self.domain.field != self.range.field:
            raise NotImplementedError('adjoint not defined since fields '
                                      'of domain and range differ ({} != {}).'
                                      ''.format(self.domain.field,
                                                self.range.field))
        return MatVecOperator(self.range, self.domain, self.matrix.H)

    def _call(self, x):
        """Raw call method on input, producing a new output."""
        return self.range.element(
            np.asarray(self.matrix.dot(x.data)).squeeze())

    def _apply(self, x, out):
        """Raw apply method on input, writing to given output."""
        if self.matrix_issparse:
            # Unfortunately, there is no native in-place dot product for
            # sparse matrices
            out.data[:] = np.asarray(self.matrix.dot(x.data)).squeeze()
        else:
            self.matrix.dot(x.data, out=out.data)

    # TODO: repr and str


def _norm_default(x):
    if _blas_is_applicable(x):
        norm = get_blas_funcs('nrm2', dtype=x.dtype)
    else:
        norm = np.linalg.norm
    return norm(x.data)


def _inner_default(x1, x2):
    if _blas_is_applicable(x1, x2):
        dot = get_blas_funcs('dotc', dtype=x1.dtype)
    elif is_real_dtype(x1.dtype):
        dot = np.dot  # still much faster than vdot
    else:
        dot = np.vdot  # slowest alternative
    # x2 as first argument because we want linearity in x1
    return dot(x2.data, x1.data)


class _FnWeighting(with_metaclass(ABCMeta, _FnWeightingBase)):

    """Abstract base class for `Fn` weighting."""


class _FnMatrixWeighting(_FnWeighting):

    """Matrix-weighting for `Fn`.

    The weighted inner product with matrix :math:`G` is defined as

    :math:`<a, b> := b^H G a`

    with :math:`b^H` standing for transposed complex conjugate. The
    matrix must be Hermitian and posivive definite, otherwise it does
    not define an inner product. This is not checked during
    initialization.

    Norm and distance are implemented as in the base class by default.
    """

    def __init__(self, matrix, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        matrix : array-like or scipy.sparse.spmatrix
            Weighting matrix of the inner product. Its shape must be
            `(n, n)`, where `n` is the size of `space`.
        dist_using_inner : bool, optional
            Calculate `dist(x, y)` as

            `sqrt(norm(x)**2 + norm(y)**2 - 2*inner(x, y).real)`

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it is not guaranteed to
            evaluate to exactly zero for equal (but not identical)
            `x` and `y`.
        """
        super().__init__(dist_using_inner)
        if isinstance(matrix, sp.sparse.spmatrix):
            self._matrix = matrix
        else:
            self._matrix = np.asmatrix(matrix)

        if self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError('matrix with shape {} not square.'
                             ''.format(self._matrix.shape))

    @property
    def matrix(self):
        """Weighting matrix of this inner product."""
        return self._matrix

    @property
    def matrix_issparse(self):
        """Whether the representing matrix is sparse or not."""
        return isinstance(self.matrix, sp.sparse.spmatrix)

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equals : bool
            `True` if `other` is an `FnMatrixWeighting` instance with
            **identical** matrix, `False` otherwise.

        See also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (isinstance(other, _FnMatrixWeighting) and
                self.matrix is other.matrix)

    def equiv(self, other):
        """Test if `other` is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            `True` if `other` is an `FnWeighting` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of this inner product's matrix with the matrix
            or constant of `other`.
        """
        # Optimization for equality
        if self == other:
            return True

        elif isinstance(other, _FnMatrixWeighting):
            if self.matrix_issparse:
                if other.matrix_issparse:
                    # Optimization for different number of nonzero elements
                    if self.matrix.nnz != other.matrix.nnz:
                        return False
                    return (self.matrix != other.matrix).nnz == 0
                else:
                    return np.array_equal(self.matrix.todense(), other.matrix)

            else:  # matrix of `self` is dense
                if other.matrix_issparse:
                    #TODO: optimize with size checks etc.
                    return np.array_equal(self.matrix, other.matrix.todense())
                else:
                    return np.array_equal(self.matrix, other.matrix)

        elif isinstance(other, _FnConstWeighting):
            return np.array_equiv(self.matrix.diagonal(), other.const)

        else:
            return False

    def matvec(self, x, out=None):
        """The matvec operation of this inner product.

        Parameters
        ----------
        x : `FnBase.Vector`
            Input vector in the matrix-vector product
        out : `FnBase.Vector`, optional
            Output vector which the result is written to

        Returns
        -------
        out : `FnBase.Vector`
            The result of the matrix-vector multiplication. If `out`
            was provided as argument, it is returned again.
        """
        if out is not None:
            if self.matrix_issparse:
                # Unfortunately, there is no native in-place dot product for
                # sparse matrices
                out.data[:] = np.asarray(self.matrix.dot(x.data))
            else:
                self.matrix.dot(x.data, out=out.data)
        else:
            out = x.space.element(
                np.asarray(self.matrix.dot(x.data)).squeeze())

        return out

    def inner(self, x1, x2):
        """Calculate the matrix-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """
        inner = _inner_default(self.matvec(x1), x2)
        if is_real_dtype(x1.dtype):
            return float(inner)
        else:
            return complex(inner)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        if self.matrix_issparse:
            inner_fstr = ('<{shape} sparse matrix, format {fmt!r}, {nnz} '
                          'stored entries>')
            fmt = self.matrix.format
            nnz = self.matrix.nnz
            if self._dist_using_inner:
                inner_fstr += ', dist_using_inner=True'
        else:
            inner_fstr = '\n{matrix!r}'
            fmt = ''
            nnz = 0
            if self._dist_using_inner:
                inner_fstr += ',\n dist_using_inner=True'
            else:
                inner_fstr += '\n'

        inner_str = inner_fstr.format(shape=self.matrix.shape, fmt=fmt,
                                      nnz=nnz, matrix=self.matrix)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return 'Weighting: matrix =\n{}'.format(self.matrix)


class _FnConstWeighting(_FnWeighting):

    """Weighting of `Fn` by a constant.

    The weighted inner product with constant `c` is defined as

    :math:`<a, b> := b^H c a`

    with :math:`b^H` standing for transposed complex conjugate.
    """

    def __init__(self, constant, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive float
            Weighting constant of the inner product.
        dist_using_inner : bool, optional
            Calculate `dist(x, y)` as

            `sqrt(norm(x)**2 + norm(y)**2 - 2*inner(x, y).real)`

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it is not guaranteed to
            evaluate to exactly zero for equal (but not identical)
            `x` and `y`.
        """
        super().__init__(dist_using_inner)
        self._const = float(constant)
        if self._const <= 0:
            raise ValueError('constant {} is not positive'.format(constant))

    @property
    def const(self):
        """Weighting constant of this inner product."""
        return self._const

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnConstWeighting`
            instance with the same constant, `False` otherwise.
        """
        # TODO: make symmetric
        return (isinstance(other, _FnConstWeighting) and
                self.const == other.const)

    def equiv(self, other):
        """Test if `other` is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            `True` if `other` is an `FnWeighting` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is the same as equality
            if `other` is an `FnConstWeighting` instance, otherwise
            by entry-wise comparison of this inner product's constant
            with the matrix of `other`.
        """
        if isinstance(other, _FnConstWeighting):
            return self == other
        elif isinstance(other, _FnWeighting):
            return other.equiv(self)
        else:
            return False

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """
        inner = self.const * _inner_default(x1, x2)
        return x1.space.field.element(inner)

    def norm(self, x):
        """Calculate the constant-weighted norm of a vector.

        Parameters
        ----------
        x1 : `Fn.Vector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the vector
        """
        return sqrt(self.const) * float(_norm_default(x))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
            The distance between the vectors
        """
        if self._dist_using_inner:
            dist_squared = (_norm_default(x1)**2 + _norm_default(x2)**2 -
                            2 * _inner_default(x1, x2).real)
            if dist_squared < 0.0:  # Compensate for numerical error
                dist_squared = 0.0
            return sqrt(self.const) * float(sqrt(dist_squared))
        else:
            return sqrt(self.const) * _norm_default(x1 - x2)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.const)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return 'Weighting: const = {:.4}'.format(self.const)


class _FnNoWeighting(_FnConstWeighting):

    """Weighting of `Fn` with constant 1.

    The unweighted inner product is defined as

    :math:`<a, b> := b^H a`

    with :math:`b^H` standing for transposed complex conjugate.
    This is the CPU implementation using NumPy.
    """

    def __init__(self, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

            norm(x-y)**2 = norm(x)**2 + norm(y)**2 - 2*inner(x, y).real

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) `x` and `y`.
        """
        super().__init__(1.0, dist_using_inner)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        if self._dist_using_inner:
            inner_str = 'dist_using_inner=True'
        else:
            inner_str = ''

        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__class__.__name__


class _FnCustomInnerProduct(_FnWeighting):

    """Custom inner product on `Fn`."""

    def __init__(self, inner, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `Fn.Vector` arguments, return a complex number and
            satisfy the following conditions for all vectors `x`,
            `y` and `z` and scalars `s`:

             - `inner(x, y) == conjugate(inner(y, x))`
             - `inner(s * x, y) == s * inner(x, y)`
             - `inner(x + z, y) == inner(x, y) + inner(z, y)`
             - `inner(x, x) == 0` (approx.) only if `x == 0`
               (approx.)

        dist_using_inner : bool, optional
            Calculate `dist` using the formula

            norm(x-y)**2 = norm(x)**2 + norm(y)**2 - 2*inner(x, y).real

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) `x` and `y`.
        """
        super().__init__(dist_using_inner)

        if not callable(inner):
            raise TypeError('inner product function {!r} not callable.'
                            ''.format(inner))
        self._inner_impl = inner

    @property
    def inner(self):
        """Custom inner product of this instance.."""
        return self._inner_impl

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnCustomInnerProduct`
            instance with the same inner product, `False` otherwise.
        """
        # TODO: make symmetric
        return (isinstance(other, _FnCustomInnerProduct) and
                self.inner == other.inner)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(self.inner)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__repr__()  # TODO: prettify?


class _FnCustomNorm(_FnWeighting):

    """Custom norm on `Fn`, removes `inner`."""

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept an `Fn.Vector`
            argument, return a `RealNumber` and satisfy the
            following properties:

            - `norm(x) >= 0`
            - `norm(x) == 0` (approx.) only if `x == 0` (approx.)
            - `norm(s * x) == abs(s) * norm(x)` for `s` scalar
            - `norm(x + y) <= norm(x) + norm(y)`
        """
        super().__init__(dist_using_inner=False)

        if not callable(norm):
            raise TypeError('norm function {!r} not callable.'
                            ''.format(norm))
        self._norm_impl = norm

    @property
    def norm(self):
        """Custom norm of this instance.."""
        return self._norm_impl

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnCustomNorm`
            instance with the same norm, `False` otherwise.
        """
        # TODO: make symmetric
        return (isinstance(other, _FnCustomNorm) and
                self.norm == other.norm)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.norm)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__repr__()  # TODO: prettify?


class _FnCustomDist(_FnWeighting):

    """Custom distance on `Fn`, removes `norm` and `inner`."""

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on :math:`F^n`.
            It must accept two `Fn.Vector` arguments and fulfill the
            following conditions for any vectors `x`, `y` and `z`:

            - `dist(x, y) == dist(y, x)`
            - `dist(x, y) >= 0`
            - `dist(x, y) == 0` (approx.) if and only if `x == y`
              (approx.)
            - `dist(x, y) <= dist(x, z) + dist(z, y)`
        """
        super().__init__(dist_using_inner=False)

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
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equal : bool
            `True` if `other` is an `FnCustomDist`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, _FnCustomDist) and
                self.dist == other.dist)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.dist)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__repr__() <==> repr(w)`."""
        return self.__repr__()  # TODO: prettify?


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
