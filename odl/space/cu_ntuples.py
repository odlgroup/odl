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

"""CUDA implementation of n-dimensional Cartesian spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from builtins import int, super
from future import standard_library
standard_library.install_aliases()
from odl.util.utility import with_metaclass

# External module imports
import numpy as np
from abc import ABCMeta

# ODL imports
from odl.space.base_ntuples import NtuplesBase, FnBase, _FnWeightingBase
from odl.util.utility import is_real_dtype, dtype_repr
import odlpp.odlpp_cuda as cuda


__all__ = ('CudaNtuples', 'CudaFn', 'CudaRn', 'CUDA_DTYPES')


def _get_int_type():
    if np.dtype(np.int).itemsize == 4:
        return 'CudaVectorInt32'
    elif np.dtype(np.int).itemsize == 8:
        return 'CudaVectorInt64'
    else:
        return 'CudaVectorIntNOT_AVAILABLE'


def _add_if_exists(dtype, name):
    if hasattr(cuda, name):
        _TYPE_MAP_NPY2CUDA[np.dtype(dtype)] = getattr(cuda, name)
        CUDA_DTYPES.append(np.dtype(dtype))


# A list of all available dtypes
CUDA_DTYPES = []

# Typemap from numpy dtype to implementations
_TYPE_MAP_NPY2CUDA = {}

# Initialize the available dtypes
_add_if_exists(np.float, 'CudaVectorFloat64')
_add_if_exists(np.float32, 'CudaVectorFloat32')
_add_if_exists(np.float64, 'CudaVectorFloat64')
_add_if_exists(np.int, _get_int_type())
_add_if_exists(np.int8, 'CudaVectorInt8')
_add_if_exists(np.int16, 'CudaVectorInt16')
_add_if_exists(np.int32, 'CudaVectorInt32')
_add_if_exists(np.int64, 'CudaVectorInt64')
_add_if_exists(np.uint8, 'CudaVectorUInt8')
_add_if_exists(np.uint16, 'CudaVectorUInt16')
_add_if_exists(np.uint32, 'CudaVectorUInt32')
_add_if_exists(np.uint64, 'CudaVectorUInt64')


class CudaNtuples(NtuplesBase):

    """The set of ``n``-tuples of arbitrary type, implemented in CUDA."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : int
            The number entries per tuple
        dtype : `object`
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Check :const:`CUDA_DTYPES` for a list of available data types.
        """

        if dtype not in _TYPE_MAP_NPY2CUDA.keys():
            raise TypeError('data type {} not supported in CUDA'.format(dtype))

        super().__init__(size, dtype)

        self._vector_impl = _TYPE_MAP_NPY2CUDA[self._dtype]

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : array-like or scalar, optional
            Input to initialize the new element.

            If ``inp`` is a `numpy.ndarray` of shape ``(size,)`` and the
            same data type as this space, the array is wrapped, not
            copied.
            Other array-like objects are copied (with broadcasting
            if necessary).

            If a single value is given, it is copied to all entries.
            TODO: make this work

        data_ptr : `int`, optional
            Memory address of a CUDA array container

        Arguments ``inp`` and `data_ptr` cannot be given at the same
        time.

        If both ``inp`` and `data_ptr` are `None`, an empty element is
        created with no guarantee of its state (memory allocation
        only).


        Returns
        -------
        element : :class:`CudaNtuples.Vector`
            The new element

        Note
        ----
        This method preserves "array views" of correct size and type,
        see the examples below.

        TODO: No, it does not yet!

        Examples
        --------
        >>> uc3 = CudaNtuples(3, 'uint8')
        >>> x = uc3.element(np.array([1, 2, 3], dtype='uint8'))
        >>> x
        CudaNtuples(3, 'uint8').element([1, 2, 3])
        >>> y = uc3.element([1, 2, 3])
        >>> y
        CudaNtuples(3, 'uint8').element([1, 2, 3])
        """
        if inp is None:
            if data_ptr is None:
                return self.Vector(self, self._vector_impl(self.size))
            else:  # TODO handle non-1 length strides
                return self.Vector(
                    self, self._vector_impl.from_pointer(data_ptr, self.size,
                                                         1))
        else:
            if data_ptr is None:
                if isinstance(inp, self._vector_impl):
                    return self.Vector(self, inp)
                else:
                    elem = self.element()
                    elem[:] = inp
                    return elem
            else:
                raise ValueError('cannot provide both inp and data_ptr.')

    class Vector(NtuplesBase.Vector):

        """Representation of a :class:`CudaNtuples` element."""

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, CudaNtuples):
                raise TypeError('{!r} not a `CudaNtuples` instance.'
                                ''.format(space))

            self._data = data

            super().__init__(space)

            if not isinstance(data, self._space._vector_impl):
                raise TypeError('data {!r} not a `{}` instance.'
                                ''.format(data, self._space._vector_impl))

        @property
        def data(self):
            """The data of this vector.

            Parameters
            ----------
            None

            Returns
            -------
            ptr : CudaFnVectorImpl
                Underlying cuda data representation

            Examples
            --------
            """
            return self._data

        @property
        def data_ptr(self):
            """A raw pointer to the data of this vector."""
            return self.data.data_ptr()

        def __eq__(self, other):
            """``vec.__eq__(other) <==> vec == other``.

            Returns
            -------
            equals : `bool`
                `True` if all elements of ``other`` are equal to this
                vector's elements, `False` otherwise

            Examples
            --------
            >>> r3 = CudaNtuples(3, 'float32')
            >>> x = r3.element([1, 2, 3])
            >>> x == x
            True
            >>> y = r3.element([1, 2, 3])
            >>> x == y
            True
            >>> y = r3.element([0, 0, 0])
            >>> x == y
            False
            >>> r3_2 = CudaNtuples(3, 'uint8')
            >>> z = r3_2.element([1, 2, 3])
            >>> x != z
            True
            """
            if other is self:
                return True
            elif other not in self.space:
                return False
            else:
                return self.data == other.data

        def copy(self):
            """Create an identical (deep) copy of this vector.

            Returns
            -------
            copy : :class:`CudaNtuples.Vector`
                The deep copy

            Examples
            --------
            >>> vec1 = CudaNtuples(3, 'uint8').element([1, 2, 3])
            >>> vec2 = vec1.copy()
            >>> vec2
            CudaNtuples(3, 'uint8').element([1, 2, 3])
            >>> vec1 == vec2
            True
            >>> vec1 is vec2
            False
            """
            return self.space.Vector(self.space, self.data.copy())

        def asarray(self, start=None, stop=None, step=None, out=None):
            """Extract the data of this array as a numpy array.

            Parameters
            ----------
            start : `int`, Optional (default: `None`)
                Start position. None means the first element.
            start : `int`, Optional (default: `None`)
                One element past the last element to be extracted.
                None means the last element.
            start : `int`, Optional (default: `None`)
                Step length. None means 1.
            out : :class:`numpy.ndarray`, Optional (default: `None`)
                Array in which the result should be written in-place.
                Has to be contiguous and of the correct dtype.

            Returns
            -------
            asarray : :class:`numpy.ndarray`
                Numpy array of the same type as the space.

            Examples
            --------
            >>> uc3 = CudaNtuples(3, 'uint8')
            >>> y = uc3.element([1, 2, 3])
            >>> y.asarray()
            array([1, 2, 3], dtype=uint8)
            >>> y.asarray(1, 3)
            array([2, 3], dtype=uint8)

            Using the out parameter

            >>> out = np.empty((3,), dtype='uint8')
            >>> result = y.asarray(out=out)
            >>> out
            array([1, 2, 3], dtype=uint8)
            >>> result is out
            True
            """
            if out is None:
                return self.data.get_to_host(slice(start, stop, step))
            else:
                self.data.copy_device_to_host(slice(start, stop, step), out)
                return out

        def __getitem__(self, indices):
            """Access values of this vector.

            This will cause the values to be copied to CPU
            which is a slow operation.

            Parameters
            ----------
            indices : `int` or `slice`
                The position(s) that should be accessed

            Returns
            -------
            values : :attr:`CudaNtuples.dtype` or :class:`CudaNtuples.Vector`
                The value(s) at the index (indices)


            Examples
            --------
            >>> uc3 = CudaNtuples(3, 'uint8')
            >>> y = uc3.element([1, 2, 3])
            >>> y[0]
            1
            >>> z = y[1:3]
            >>> z
            CudaNtuples(2, 'uint8').element([2, 3])
            >>> y[::2]
            CudaNtuples(2, 'uint8').element([1, 3])
            >>> y[::-1]
            CudaNtuples(3, 'uint8').element([3, 2, 1])

            The returned value is a view, modifications are reflected
            in the original data:

            >>> z[:] = [4, 5]
            >>> y
            CudaNtuples(3, 'uint8').element([1, 4, 5])
            """
            if isinstance(indices, slice):
                data = self.data.getslice(indices)
                return type(self.space)(data.size, data.dtype).element(data)
            else:
                return self.data.__getitem__(indices)

        def __setitem__(self, indices, values):
            """Set values of this vector.

            This will cause the values to be copied to CPU
            which is a slow operation.

            Parameters
            ----------
            indices : `int` or `slice`
                The position(s) that should be set
            values : {scalar, array-like, :class:`CudaNtuples.Vector`}
                The value(s) that are to be assigned.

                If ``index`` is an `int`, ``value`` must be single value.

                If ``index`` is a `slice`, ``value`` must be broadcastable
                to the size of the slice (same size, shape (1,)
                or single value).

            Returns
            -------
            None

            Examples
            --------
            >>> uc3 = CudaNtuples(3, 'uint8')
            >>> y = uc3.element([1, 2, 3])
            >>> y[0] = 5
            >>> y
            CudaNtuples(3, 'uint8').element([5, 2, 3])
            >>> y[1:3] = [7, 8]
            >>> y
            CudaNtuples(3, 'uint8').element([5, 7, 8])
            >>> y[:] = np.array([0, 0, 0])
            >>> y
            CudaNtuples(3, 'uint8').element([0, 0, 0])

            Scalar assignment

            >>> y[:] = 5
            >>> y
            CudaNtuples(3, 'uint8').element([5, 5, 5])
            """
            if isinstance(values, CudaNtuples.Vector):
                self.assign(values)  # use lincomb magic
            else:
                if isinstance(indices, slice):
                    # Convert value to the correct type if needed
                    value_array = np.asarray(values, dtype=self.space.dtype)

                    if (value_array.ndim == 0):
                        self.data.fill(values)
                    else:
                        # Size checking is performed in c++
                        self.data.setslice(indices, value_array)
                else:
                    self.data.__setitem__(int(indices), values)


class CudaFn(FnBase, CudaNtuples):

    """The space :math:`F^n`, implemented in CUDA.

    Requires the compiled ODL extension odlpp.
    """

    def __init__(self, size, dtype, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only scalar data types are allowed.

        kwargs : {'weight', 'dist', 'norm', 'inner'}
            'weight' : `float` or `None`
                Use weighted inner product, norm, and dist.

                `None` (default) : Use the standard unweighted functions

                `float` : Use functions weighted by a constant

                This option cannot be combined with ``dist``, ``norm`` or
                ``inner``.

            'dist' : callable, optional
                The distance function defining a metric on :math:`F^n`.
                It must accept two :class:`CudaFn.Vector` arguments and fulfill
                the following conditions for any vectors ``x``, ``y`` and
                ``z``:

                - ``dist(x, y) == dist(y, x)``
                - ``dist(x, y) >= 0``
                - ``dist(x, y) == 0`` (approx.) if and only if ``x == y``
                  (approx.)
                - ``dist(x, y) <= dist(x, z) + dist(z, y)``

                This option cannot be combined with ``weight``, ``norm``
                or ``inner``.

            'norm' : callable, optional
                The norm implementation. It must accept a
                :class:`CudaFn.Vector` argument, return a real number and
                satisfy the following properties:

                - ``norm(x) >= 0``
                - ``norm(x) == 0` (approx.) only if `x == 0`` (approx.)
                - ``norm(s * x) == abs(s) * norm(x)`` for ``s`` scalar
                - ``norm(x + y) <= norm(x) + norm(y)``

                This option cannot be combined with ``weight``, ``dist``
                or ``inner``.

            'inner' : callable, optional
                The inner product implementation. It must accept two
                :class:`CudaFn.Vector` arguments, return a `complex` number and
                satisfy the following conditions for all vectors ``x``,
                ``y`` and ``z`` and scalars ``s``:

                 - ``inner(x, y) == conjugate(inner(y, x))``
                 - ``inner(s * x, y) == s * inner(x, y)``
                 - ``inner(x + z, y) == inner(x, y) + inner(z, y)``
                 - ``inner(x, x) == 0`` (approx.) only if ``x == 0``
                   (approx.)

                This option cannot be combined with ``weight``, ``dist``
                or ``norm``.
        """
        super().__init__(size, dtype)
        CudaNtuples.__init__(self, size, dtype)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if (dist, norm, inner, weight).count(None) < 3:
            raise ValueError('invalid combination of options `weight`, '
                             '`dist`, `norm` and `inner`.')
        if weight is not None:
            if np.isscalar(weight):
                self._space_funcs = _CudaFnConstWeighting(weight)
            elif weight is None:
                pass
            else:
                raise ValueError('invalid weight argument {!r}.'
                                 ''.format(weight))
        elif dist is not None:
            self._space_funcs = _CudaFnCustomDist(dist)
        elif norm is not None:
            self._space_funcs = _CudaFnCustomNorm(norm)
        elif inner is not None:
            # Use fast dist implementation
            self._space_funcs = _CudaFnCustomInnerProduct(
                inner, dist_using_inner=True)
        else:  # all None -> no weighing
            self._space_funcs = _CudaFnNoWeighting()

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of ``x1`` and ``x2``, assigned to ``out``.

        Calculate `z = a * x + b * y` using optimized CUDA routines.

        Parameters
        ----------
        a, b : :attr:`field` element
            Scalar to multiply ``x`` and ``y`` with.
        x, y : :class:`CudaFn.Vector`
            The summands
        out : :class:`CudaFn.Vector`
            The Vector that the result is written to.

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = CudaFn(3, 'float32')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> out = r3.element()
        >>> r3.lincomb(2, x, 3, y, out)  # out is returned
        CudaFn(3, 'float32').element([14.0, 19.0, 24.0])
        >>> out
        CudaFn(3, 'float32').element([14.0, 19.0, 24.0])
        """
        out.data.lincomb(a, x1.data, b, x2.data)

    def _inner(self, x1, x2):
        """Calculate the inner product of x and y.

        Parameters
        ----------
        x1, x2 : :class:`CudaFn.Vector`

        Returns
        -------
        inner: `float` or `complex`
            The inner product of x and y


        Examples
        --------
        >>> uc3 = CudaFn(3, 'uint8')
        >>> x = uc3.element([1, 2, 3])
        >>> y = uc3.element([3, 1, 5])
        >>> uc3.inner(x, y)
        20.0
        """
        return self._space_funcs.inner(x1, x2)

    def _dist(self, x1, x2):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x1, x2 : :class:`CudaFn.Vector`
            The vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
            Distance between the vectors

        Examples
        --------
        >>> r2 = CudaRn(2)
        >>> x = r2.element([3, 8])
        >>> y = r2.element([0, 4])
        >>> r2.dist(x, y)
        5.0
        """
        return self._space_funcs.dist(x1, x2)

    def _norm(self, x):
        """Calculate the norm of ``x``.

        This method is implemented separately from ``sqrt(inner(x,x))``
        for efficiency reasons.

        Parameters
        ----------
        x : :class:`CudaFn.Vector`

        Returns
        -------
        norm : `float`
            The norm of x

        Examples
        --------
        >>> uc3 = CudaFn(3, 'uint8')
        >>> x = uc3.element([2, 3, 6])
        >>> uc3.norm(x)
        7.0
        """
        return self._space_funcs.norm(x)

    def _multiply(self, x1, x2, out):
        """The pointwise product of two vectors, assigned to ``out``.

        This is defined as:

        multiply(x, y, out) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

        Parameters
        ----------

        x1, x2 : CudaFn.Vector
            Factors in product
        out : CudaFn.Vector
            Result

        Returns
        -------
        None

        Examples
        --------

        >>> rn = CudaRn(3)
        >>> x1 = rn.element([5, 3, 2])
        >>> x2 = rn.element([1, 2, 3])
        >>> out = rn.element()
        >>> rn.multiply(x1, x2, out)  # out is returned
        CudaRn(3).element([5.0, 6.0, 6.0])
        >>> out
        CudaRn(3).element([5.0, 6.0, 6.0])
        """
        out.data.multiply(x1.data, x2.data)

    def _divide(self, x1, x2, out):
        """The pointwise division of two vectors, assigned to ``out``.

        This is defined as:

        multiply(z, x, y) := [x[0]/y[0], x[1]/y[1], ..., x[n-1]/y[n-1]]

        Parameters
        ----------

        x1, x2 : CudaFn.Vector
            Read from
        out : CudaFn.Vector
            Write to

        Returns
        -------
        None

        Examples
        --------

        >>> rn = CudaRn(3)
        >>> x1 = rn.element([5, 3, 2])
        >>> x2 = rn.element([1, 2, 2])
        >>> out = rn.element()
        >>> rn.divide(x1, x2, out)  # out is returned
        CudaRn(3).element([5.0, 1.5, 1.0])
        >>> out
        CudaRn(3).element([5.0, 1.5, 1.0])
        """
        out.data.divide(x1.data, x2.data)

    def zero(self):
        """Create a vector of zeros."""
        return self.Vector(self, self._vector_impl(self.size, 0))

    def one(self):
        """Create a vector of ones."""
        return self.Vector(self, self._vector_impl(self.size, 1))

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_fstr = '{}, {}'
        weight = 1.0
        if self._space_funcs._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'
        if isinstance(self._space_funcs, _CudaFnCustomInnerProduct):
            inner_fstr += ', inner=<custom inner>'
        elif isinstance(self._space_funcs, _CudaFnCustomNorm):
            inner_fstr += ', norm=<custom norm>'
        elif isinstance(self._space_funcs, _CudaFnCustomDist):
            inner_fstr += ', norm=<custom dist>'
        elif isinstance(self._space_funcs, _CudaFnConstWeighting):
            weight = self._space_funcs.const
            if weight != 1.0:
                inner_fstr += ', weight={weight}'

        inner_str = inner_fstr.format(self.size, dtype_repr(self.dtype),
                                      weight=weight)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    class Vector(FnBase.Vector, CudaNtuples.Vector):

        """Representation of a :class:`CudaFn` element.

        # TODO: document public interface
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            CudaNtuples.Vector.__init__(self, space, data)


class CudaRn(CudaFn):

    """The real space :math:`R^n`, implemented in CUDA.

    Requires the compiled ODL extension odlpp.
    """

    def __init__(self, size, dtype=np.float32, **kwargs):
        """Initialize a new instance.

        Parameters
        ----------
        size : positive int
            The number of dimensions of the space
        dtype : object
            The data type of the storage array. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only real floating-point data types are allowed.

        kwargs : {'weight', 'dist', 'norm', 'inner'}
            See :class:`CudaFn`
        """
        super().__init__(size, dtype, **kwargs)

        if not is_real_dtype(self._dtype):
            raise TypeError('data type {} not a real floating-point type.'
                            ''.format(dtype))

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        inner_fstr = '{}'
        weight = 1.0
        if self.dtype != np.float32:
            inner_fstr += ', {dtype}'
        if self._space_funcs._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'
        if isinstance(self._space_funcs, _CudaFnCustomInnerProduct):
            inner_fstr += ', inner=<custom inner>'
        elif isinstance(self._space_funcs, _CudaFnCustomNorm):
            inner_fstr += ', norm=<custom norm>'
        elif isinstance(self._space_funcs, _CudaFnCustomDist):
            inner_fstr += ', norm=<custom dist>'
        elif isinstance(self._space_funcs, _CudaFnConstWeighting):
            weight = self._space_funcs.const
            if weight != 1.0:
                inner_fstr += ', weight={weight}'

        inner_str = inner_fstr.format(self.size, dtype_repr(self.dtype),
                                      weight=weight)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    class Vector(CudaFn.Vector):
        pass


# Methods
# TODO: move

def _make_unary_fun(name):
    def fun(x, out=None):
        if out is None:
            out = x.space.element()
        getattr(x.data, name)(out.data)
        return out
    return fun

sin = _make_unary_fun('sin')
cos = _make_unary_fun('cos')
arcsin = _make_unary_fun('arcsin')
arccos = _make_unary_fun('arccos')
log = _make_unary_fun('log')
exp = _make_unary_fun('exp')
abs = _make_unary_fun('abs')
sign = _make_unary_fun('sign')
sqrt = _make_unary_fun('sqrt')


def add_scalar(x, scal, out=None):
    if out is None:
        out = x.space.element()
    cuda.add_scalar(x.data, scal, out.data)
    return out


def max_vector_scalar(x, scal, out=None):
    if out is None:
        out = x.space.element()
    cuda.max_vector_scalar(x.data, scal, out.data)
    return out


def max_vector_vector(x1, x2, out=None):
    if out is None:
        out = x1.space.element()
    cuda.max_vector_vector(x1.data, x2.data, out.data)
    return out


def divide_vector_vector(x1, x2, out=None):
    if out is None:
        out = x1.space.element()
    cuda.divide_vector_vector(x1.data, x2.data, out.data)
    return out


def sum(x):
    return cuda.sum(x.data)


def _dist_default(x1, x2):
    return x1.data.dist(x2.data)


def _norm_default(x):
    return x.data.norm()


def _inner_default(x1, x2):
    return x1.data.inner(x2.data)


class _CudaFnWeighting(with_metaclass(ABCMeta, _FnWeightingBase)):

    """Abstract base class for :class:`CudaFn` weighting."""


class _CudaFnConstWeighting(_CudaFnWeighting):

    """Weighting of :class:`CudaFn` by a constant.

    The weighted inner product with constant ``c`` is defined as

    :math:`<a, b> := b^H c a`

    with :math:`b^H` standing for transposed `complex` conjugate.
    """

    def __init__(self, constant):
        """Initialize a new instance.

        Parameters
        ----------
        constant : `float`
            Weighting constant of the inner product.
        """
        super().__init__(dist_using_inner=False)
        self._const = float(constant)

    @property
    def const(self):
        """Weighting constant of this inner product."""
        return self._const

    def __eq__(self, other):
        """``inner.__eq__(other) <==> inner == other``.

        Returns
        -------
        equal : `bool`
            `True` if ``other`` is a `_CudaFnConstWeighting`
            instance with the same constant, `False` otherwise.
        """
        return (isinstance(other, _CudaFnConstWeighting) and
                self.const == other.const)

    def equiv(self, other):
        """Test if ``other`` is an equivalent weighting.

        Returns
        -------
        equivalent : `bool`
            `True` if ``other`` is a `_CudaFnWeighting` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is the same as equality
            if ``other`` is a `_CudaFnConstWeighting` instance, otherwise
            by entry-wise comparison of this inner product's constant
            with the matrix of ``other``.
        """
        if isinstance(other, _CudaFnConstWeighting):
            return self == other
        elif isinstance(other, _CudaFnWeighting):
            return other.equiv(self)
        else:
            return False

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : :class:`CudaFn.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : `float` or `complex`
            The inner product of the two provided vectors
        """
        return self.const * float(_inner_default(x1, x2))

    def norm(self, x):
        """Calculate the constant-weighted norm of a vector.

        Parameters
        ----------
        x1 : :class:`CudaFn.Vector`
            Vector whose norm is calculated

        Returns
        -------
        norm : `float`
            The norm of the vector
        """
        from math import sqrt
        return sqrt(self.const) * float(_norm_default(x))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : :class:``CudaFn.Vector``
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
            The distance between the vectors
        """
        from math import sqrt
        from builtins import abs
        return sqrt(abs(self.const)) * float(_dist_default(x1, x2))

    def __repr__(self):
        """``w.__repr__() <==> repr(w)``."""
        inner_fstr = '{}'
        inner_str = inner_fstr.format(self.const)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """``w.__str__() <==> str(w)``."""
        return '{}: constant = {:.4}'.format(self.__class__.__name__,
                                             self.const)


class _CudaFnNoWeighting(_CudaFnConstWeighting):

    """Weighting of :class:`CudaFn` with constant 1.

    The unweighted inner product is defined as

    :math:`<a, b> := b^H a`

    with :math:`b^H` standing for transposed `complex` conjugate.
    This is the CPU implementation using NumPy.
    """

    def __init__(self):
        """Initialize a new instance."""
        super().__init__(1.0)

    def __repr__(self):
        """``w.__repr__() <==> repr(w)``."""
        return '{}()'.format(self.__class__.__name__)

    def __str__(self):
        """``w.__str__() <==> str(w)``."""
        return self.__class__.__name__


class _CudaFnCustomInnerProduct(_CudaFnWeighting):

    """Custom inner product on :class:`CudaFn`."""

    def __init__(self, inner, dist_using_inner=False):
        """Initialize a new instance.

        Parameters
        ----------
        inner : `callable`
            The inner product implementation. It must accept two
            :class:`CudaFn.Vector` arguments, return a `complex` number and
            satisfy the following conditions for all vectors ``x``,
            ``y`` and ``z`` and scalars ``s``:

             - ``inner(x, y) == conjugate(inner(y, x))``
             - ``inner(s * x, y) == s * inner(x, y)``
             - ``inner(x + z, y) == inner(x, y) + inner(z, y)``
             - ``inner(x, x) == 0`` (approx.) only if ``x == 0``
               (approx.)

        dist_using_inner : `bool`, optional
            Calculate ``dist`` using the formula

            ``norm(x-y)**2 = norm(x)**2 + norm(y)**2 - 2*inner(x, y).real``

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.
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
        """``inner.__eq__(other) <==> inner == other``.

        Returns
        -------
        equal : `bool`
            `True` if ``other`` is an :class:``CudaFnCustomInnerProduct``
            instance with the same inner product, `False` otherwise.
        """
        return (isinstance(other, _CudaFnCustomInnerProduct) and
                self.inner == other.inner)

    def __repr__(self):
        """``w.__repr__() <==> repr(w)``."""
        inner_fstr = '{!r}'
        if self._dist_using_inner:
            inner_fstr += ',dist_using_inner={dist_u_i}'

        inner_str = inner_fstr.format(self.inner,
                                      dist_u_i=self._dist_using_inner)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """``w.__str__() <==> str(w)``."""
        return self.__repr__()  # TODO: prettify?


class _CudaFnCustomNorm(_CudaFnWeighting):

    """Custom norm on :class:`CudaFn`, removes ``inner``."""

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : `callable`
            The norm implementation. It must accept an :class:``CudaFn.Vector``
            argument, return a `float` and satisfy the
            following properties:

            - ``norm(x) >= 0``
            - ``norm(x) == 0`` (approx.) only if ``x == 0`` (approx.)
            - ``norm(s * x) == abs(s) * norm(x)`` for ``s`` scalar
            - ``norm(x + y) <= norm(x) + norm(y)``
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
        """``inner.__eq__(other) <==> inner == other``.

        Returns
        -------
        equal : `bool`
            `True` if ``other`` is an `CudaFnCustomNorm`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, _CudaFnCustomNorm) and
                self.norm == other.norm)

    def __repr__(self):
        """``w.__repr__() <==> repr(w)``."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.norm)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """``w.__str__() <==> str(w)``."""
        return self.__repr__()  # TODO: prettify?


class _CudaFnCustomDist(_CudaFnWeighting):

    """Custom distance on :class:`CudaFn`, removes ``norm`` and ``inner``."""

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : `callable`
            The distance function defining a metric on :math:`F^n`.
            It must accept two :class:``CudaFn.Vector`` arguments and fulfill the
            following conditions for any vectors ``x``, ``y`` and ``z``:

            - ``dist(x, y) == dist(y, x)``
            - ``dist(x, y) >= 0``
            - ``dist(x, y) == 0`` (approx.) if and only if ``x == y``
              (approx.)
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``
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
        """``inner.__eq__(other) <==> inner == other``.

        Returns
        -------
        equal : `bool`
            `True` if ``other`` is an `CudaFnCustomDist`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, _CudaFnCustomDist) and
                self.dist == other.dist)

    def __repr__(self):
        """``w.__repr__() <==> repr(w)``."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.dist)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """``w.__str__() <==> str(w)``."""
        return self.__repr__()  # TODO: prettify?


import os
if not os.environ.get('READTHEDOCS', None) == 'True':
    try:
        CudaRn(1).element()
    except (MemoryError, RuntimeError, TypeError) as err:
        print(err)
        raise ImportError('Your GPU seems to be misconfigured. Skipping '
                          'CUDA-dependent modules.')


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
