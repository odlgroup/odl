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
from future.utils import with_metaclass

# External module imports
import numpy as np
from abc import ABCMeta

# ODL imports
from odl.space.ntuples import NtuplesBase, FnBase
from odl.space.ntuples import WeightedInnerBase, ConstWeightedInner
import odlpp.odlpp_cuda as cuda


__all__ = ('CudaNtuples', 'CudaFn', 'CudaRn', 'CudaConstWeightedInner')

def _get_int_type():
    if np.dtype(np.int).itemsize == 4:
        return 'CudaVectorInt32'
    elif np.dtype(np.int).itemsize == 8:
        return 'CudaVectorInt64'
    else:
        raise NotImplementedError("int size not implemented")
    
def _add_if_exists(dtype, name):
    if hasattr(cuda, name):
        _TYPE_MAP_NPY2CUDA[np.dtype(dtype)] = getattr(cuda, name)
        AVAILABLE_DTYPES.append(np.dtype(dtype))


#A list of all available dtypes
AVAILABLE_DTYPES = []

#Typemap from numpy dtype to implementations
_TYPE_MAP_NPY2CUDA = {}

#Initialize the available dtypes
_add_if_exists(np.float, 'CudaVectorFloat64')
_add_if_exists(np.float32, 'CudaVectorFloat32')
_add_if_exists(np.float64, 'CudaVectorFloat64')
_add_if_exists(np.int, _get_int_type())
_add_if_exists(np.int8, 'CudaVectorInt8')
_add_if_exists(np.int16, 'CudaVectorInt16')
_add_if_exists(np.int32, 'CudaVectorInt32')
_add_if_exists(np.int64, 'CudaVectorInt64')
_add_if_exists(np.uint8, 'CudaVectorUInt8')
_add_if_exists(np.uint16, 'CudaVectorInt64')
_add_if_exists(np.uint32, 'CudaVectorUInt16')
_add_if_exists(np.uint64, 'CudaVectorUInt64')

class CudaNtuples(NtuplesBase):

    """The set of `n`-tuples of arbitrary type, implemented in CUDA."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : int
            The number entries per tuple
        dtype : `object`
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Currently supported: 'float32', 'uint8'
        """
        super().__init__(size, dtype)
        if self._dtype not in _TYPE_MAP_NPY2CUDA.keys():
            raise TypeError('data type {} not supported in CUDA'.format(dtype))

        self._vector_impl = _TYPE_MAP_NPY2CUDA[self._dtype]

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : array-like or scalar, optional
            Input to initialize the new element.

            If `inp` is a `numpy.ndarray` of shape `(size,)` and the
            same data type as this space, the array is wrapped, not
            copied.
            Other array-like objects are copied (with broadcasting
            if necessary).

            If a single value is given, it is copied to all entries.
            TODO: make this work

        data_ptr : `int`, optional
            Memory address of a CUDA array container

        Arguments `inp` and `data_ptr` cannot be given at the same
        time.

        If both `inp` and `data_ptr` are `None`, an empty element is
        created with no guarantee of its state (memory allocation
        only).


        Returns
        -------
        element : `CudaNtuples.Vector`
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
                raise TypeError('cannot provide both inp and data_ptr.')

    class Vector(NtuplesBase.Vector):

        """Representation of a `CudaNtuples` element."""

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

        @property
        def nbytes(self):
            """The number of bytes this vector uses in memory."""
            return self.data.nbytes

        @property
        def itemsize(self):
            """The size in bytes on one element of this type."""
            return self.data.itemsize

        def __eq__(self, other):
            """`vec.__eq__(other) <==> vec == other`.

            Returns
            -------
            equals : `bool`
                `True` if all elements of `other` are equal to this
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
                return self.data.equals(other.data)

        def copy(self):
            """Create an identical (deep) copy of this vector.

            Returns
            -------
            copy : `CudaNtuples.Vector`
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
            out : `ndarray`, Optional (default: `None`)
                Array in which the result should be written in-place.
                Has to be contiguous and of the correct dtype.

            Returns
            -------
            asarray : `ndarray`
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
            values : `space.dtype` or `space.Vector`
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
            values : {scalar, array-like, `CudaNtuples.Vector`}
                The value(s) that are to be assigned.

                If `index` is an `int`, `value` must be single value.

                If `index` is a `slice`, `value` must be broadcastable
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
                    value_array = np.asarray(values, dtype=self.space._dtype)

                    if (value_array.ndim == 0):
                        self.data.fill(values)
                    else:
                        # Size checking is performed in c++
                        self.data.setslice(indices, value_array)
                else:
                    self.data.__setitem__(int(indices), values)


class CudaFn(FnBase, CudaNtuples):

    """The space F^n, implemented in CUDA.

    Requires the compiled ODL extension odlpp.
    """

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : int
            The number entries per tuple
        dtype : object
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only scalar data types (numbers) are allowed.

            Currently supported: 'float32', 'uint8'
        """
        super().__init__(size, dtype)
        CudaNtuples.__init__(self, size, dtype)

    def _lincomb(self, z, a, x, b, y):
        """Linear combination of `x` and `y`.

        Calculate `z = a * x + b * y` using optimized BLAS routines if
        possible.

        Parameters
        ----------
        z : `CudaFn.Vector`
            The Vector that the result is written to.
        a, b : `field` element
            Scalar to multiply `x` and `y` with.
        x, y : `CudaFn.Vector`
            The summands

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = CudaFn(3, 'float32')
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> z = r3.element()
        >>> r3.lincomb(z, 2, x, 3, y)
        >>> z
        CudaFn(3, 'float32').element([14.0, 19.0, 24.0])
        """
        z.data.lincomb(a, x.data, b, y.data)

    def _inner(self, x, y):
        """Calculate the inner product of x and y.

        Parameters
        ----------
        x, y : `CudaFn.Vector`

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
        return x.data.inner(y.data)

    def _dist(self, x, y):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x, y : `CudaFn.Vector`
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
        return x.data.dist(y.data)

    def _norm(self, x):
        """Calculate the 2-norm of x.

        This method is implemented separately from `sqrt(inner(x,x))`
        for efficiency reasons.

        Parameters
        ----------
        x : `CudaFn.Vector`

        Returns
        -------
        norm : `float`
            The 2-norm of x


        Examples
        --------
        >>> uc3 = CudaFn(3, 'uint8')
        >>> x = uc3.element([2, 3, 6])
        >>> uc3.norm(x)
        7.0
        """
        return x.data.norm()

    def _multiply(self, z, x, y):
        """The pointwise product of two vectors, assigned to `y`.

        This is defined as:

        multiply(z, x, y) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

        Parameters
        ----------

        z : CudaRn.Vector
            Write to
        x : CudaRn.Vector
            Read from
        y : CudaRn.Vector
            Read from

        Returns
        -------
        None

        Examples
        --------

        >>> rn = CudaRn(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> z = rn.element()
        >>> rn.multiply(z, x, y)
        >>> z
        CudaRn(3, 'float32').element([5.0, 6.0, 6.0])
        """
        z.data.multiply(x.data, y.data)

    def zero(self):
        """Create a vector of zeros."""
        return self.Vector(self, self._vector_impl(self.size, 0))

    class Vector(FnBase.Vector, CudaNtuples.Vector):

        """Representation of a `CudaFn` element.

        # TODO: document public interface
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            super().__init__(space, data)
            if not isinstance(data, self._space._vector_impl):
                return TypeError('data {!r} is not an instance of '
                                 '{}.'.format(data, self._space._vector_impl))


class CudaRn(CudaFn):

    """The real space :math:`R^n`, implemented in CUDA.

    Requires the compiled ODL extension odlpp.

    # TODO: document public interface
    """

    def __init__(self, size, dtype=np.float32):
        """Initialize a new instance.

        Only real floating-point types are allowed.
        """
        super().__init__(size, dtype)

        if not np.isrealobj(np.empty(0, dtype=self._dtype)):
            raise TypeError('data type {} not a real floating-point type.'
                            ''.format(dtype))

    class Vector(CudaFn.Vector):
        pass


# Methods
# TODO: move

def _make_unary_fun(name):
    def fun(inp, outp=None):
        if outp is None:
            outp = inp.space.element()
        getattr(inp.data, name)(outp.data)
        return outp
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

def add_scalar(inp, scal, outp=None):
    if outp is None:
        outp = inp.space.element()
    cuda.add_scalar(inp.data, scal, outp.data)
    return outp


def max_vector_scalar(inp, scal, outp=None):
    if outp is None:
        outp = inp.space.element()
    cuda.max_vector_scalar(inp.data, scal, outp.data)
    return outp


def max_vector_vector(inp1, inp2, outp=None):
    if outp is None:
        outp = inp1.space.element()
    cuda.max_vector_vector(inp1.data, inp2.data, outp.data)
    return outp


def divide_vector_vector(inp1, inp2, outp=None):
    if outp is None:
        outp = inp1.space.element()
    cuda.divide_vector_vector(inp1.data, inp2.data, outp.data)
    return outp


def sum(inp):
    return cuda.sum(inp.data)


class CudaWeightedInner(with_metaclass(ABCMeta, WeightedInnerBase)):

    """Abstract base class for CUDA weighted inner products. """

    def __call__(self, x, y):
        """`inner.__call__(x, y) <==> inner(x, y).`

        Calculate the inner product of `x` and `y` weighted by the
        matrix of this instance.

        Parameters
        ----------
        x, y : `CudaFn.Vector`
            Arrays whose inner product is calculated. They must have
            equal length.

        Returns
        -------
        inner : float or complex
            Weighted inner product. The output type depends on the
            input array dtype and the weighting.
        """
        if not isinstance(x, CudaFn.Vector):
            raise TypeError('x vector {!r} not a `CudaFn.Vector` instance.'
                            ''.format(x))
        if not isinstance(y, CudaFn.Vector):
            raise TypeError('y vector {!r} not a `CudaFn.Vector` instance.'
                            ''.format(y))

        if x.size != y.size:
            raise TypeError('vector sizes {} and {} are different.'
                            ''.format(x.size, y.size))

        # TODO: possibly adapt syntax once complex vectors are supported
        return self.matvec(x).inner(y)


class CudaMatrixWeightedInner(CudaWeightedInner):

    """Function object for matrix-weighted :math:`F^n` inner products.

    The weighted inner product with matrix :math:`G` is defined as

    :math:`<a, b> := b^H G a`

    with :math:`b^H` standing for transposed complex conjugate.
    """

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`."""
        raise NotImplementedError

    def matvec(self, vec):
        """Return product of the weighting matrix with a vector.

        Parameters
        ----------
        vec : CudaVector
            Array with which to multiply the weighting matrix

        Returns
        -------
        weighted : CudaVector
            The matrix-vector product as a CUDA vector
        """
        raise NotImplementedError


class CudaConstWeightedInner(CudaWeightedInner, ConstWeightedInner):

    """Constant-weighted :math:`F^n` inner product in CUDA."""

    def __repr__(self):
        """`inner.__repr__() <==> repr(inner)`."""
        return 'CudaConstWeightedInner({!r}, {})'.format(self.matvec.domain,
                                                         self.const)


try:
    CudaRn(1).element()
except MemoryError:
    raise ImportError('Warning: Your GPU seems to be misconfigured. Skipping '
                      'CUDA-dependent modules.')


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
