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
from future import standard_library
standard_library.install_aliases()
from builtins import int, super

# External module imports
import numpy as np

# ODL imports
from odl.space.base_ntuples import NtuplesBase, FnBase, _FnWeightingBase
from odl.util.utility import is_real_dtype, dtype_repr, with_metaclass
import odlpp.odlpp_cuda as cuda


__all__ = ('CudaNtuples', 'CudaFn', 'CudaRn', 'CUDA_DTYPES',
           'CudaFnConstWeighting', 'CudaFnVectorWeighting',
           'cu_weighted_inner', 'cu_weighted_norm', 'cu_weighted_dist')


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
CUDA_DTYPES = tuple(set(CUDA_DTYPES))  # Remove duplicates


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

            Check `CUDA_DTYPES` for a list of available data types.
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

            If `inp` is a `numpy.ndarray` of shape `(size,)` and the
            same data type as this space, the array is wrapped, not
            copied.
            Other array-like objects are copied (with broadcasting
            if necessary).

            If a single value is given, it is copied to all entries.

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
            else:  # TODO: handle non-1 length strides
                return self.Vector(
                    self, self._vector_impl.from_pointer(data_ptr, self.size,
                                                         1))
        else:
            if data_ptr is None:
                if isinstance(inp, self._vector_impl):
                    return self.Vector(self, inp)
                elif isinstance(inp, self.Vector) and inp.dtype == self.dtype:
                    return self.Vector(self, inp.data)
                else:
                    elem = self.element()
                    elem[:] = inp
                    return elem
            else:
                raise ValueError('cannot provide both inp and data_ptr.')

    class Vector(NtuplesBase.Vector):

        """Representation of a `CudaNtuples` element."""

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, CudaNtuples):
                raise TypeError('{!r} not a `CudaNtuples` instance.'
                                ''.format(space))

            super().__init__(space)

            if not isinstance(data, self._space._vector_impl):
                raise TypeError('data {!r} not a `{}` instance.'
                                ''.format(data, self._space._vector_impl))
            self._data = data

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
                return self.data == other.data

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
                    value_array = np.asarray(values, dtype=self.space.dtype)

                    if (value_array.ndim == 0):
                        self.data.fill(values)
                    else:
                        # Size checking is performed in c++
                        self.data.setslice(indices, value_array)
                else:
                    self.data.__setitem__(int(indices), values)


def _repr_space_funcs(space):
    inner_str = ''

    weight = 1.0
    if space._space_funcs._dist_using_inner:
        inner_str += ', dist_using_inner=True'
    if isinstance(space._space_funcs, _CudaFnCustomInnerProduct):
        inner_str += ', inner=<custom inner>'
    elif isinstance(space._space_funcs, _CudaFnCustomNorm):
        inner_str += ', norm=<custom norm>'
    elif isinstance(space._space_funcs, _CudaFnCustomDist):
        inner_str += ', norm=<custom dist>'
    elif isinstance(space._space_funcs, CudaFnConstWeighting):
        weight = space._space_funcs.const
        if weight != 1.0:
            inner_str += ', weight={}'.format(weight)

    exponent = space._space_funcs.exponent
    if exponent != 2.0:
        inner_str += ', exponent={}'.format(exponent)

    return inner_str


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
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only scalar data types are allowed.

        kwargs : {'weight', 'exponent', 'dist', 'norm', 'inner'}
            'weight' : array-like, `CudaFn.Vector`, float or `None`
                Use weighted inner product, norm, and dist.

                `None` (default) : No weighting, use standard functions

                float : Weighting by a constant

                array-like : Weighting by a vector (1-dim. array,
                corresponds to a diagonal matrix). Note that the array
                is stored in main memory, which results in slower
                space functions due to a copy during evaluation.

                `CudaFn.Vector` : same as 1-dim. array-like, except
                that copying is avoided if the `dtype` of the vector
                is the same as this space's `dtype`.

                This option cannot be combined with `dist`, `norm` or
                `inner`.

            'exponent' : positive float
                Exponent of the norm. For values other than 2.0, no
                inner product is defined.

                This option is ignored if `dist`, `norm` or `inner`
                is given.

                Default: 2.0

            'dist' : callable, optional
                The distance function defining a metric on :math:`F^n`.
                It must accept two `CudaFn.Vector` arguments and fulfill
                the following conditions for any vectors `x`, `y` and
                `z`:

                - `dist(x, y) == dist(y, x)`
                - `dist(x, y) >= 0`
                - `dist(x, y) == 0` (approx.) if and only if `x == y`
                  (approx.)
                - `dist(x, y) <= dist(x, z) + dist(z, y)`

                This option cannot be combined with `weight`, `norm`
                or `inner`.

            'norm' : callable, optional
                The norm implementation. It must accept a
                `CudaFn.Vector` argument, return a real number and
                satisfy the following properties:

                - `norm(x) >= 0`
                - `norm(x) == 0` (approx.) only if `x == 0` (approx.)
                - `norm(s * x) == abs(s) * norm(x)` for `s` scalar
                - `norm(x + y) <= norm(x) + norm(y)`

                This option cannot be combined with `weight`, `dist`
                or `inner`.

            'inner' : callable, optional
                The inner product implementation. It must accept two
                `CudaFn.Vector` arguments, return a complex number and
                satisfy the following conditions for all vectors `x`,
                `y` and `z` and scalars `s`:

                 - `inner(x, y) == conjugate(inner(y, x))`
                 - `inner(s * x, y) == s * inner(x, y)`
                 - `inner(x + z, y) == inner(x, y) + inner(z, y)`
                 - `inner(x, x) == 0` (approx.) only if `x == 0`
                   (approx.)

                This option cannot be combined with `weight`, `dist`
                or `norm`.
        """
        super().__init__(size, dtype)
        CudaNtuples.__init__(self, size, dtype)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)
        exponent = kwargs.pop('exponent', 2.0)

        # Check validity of option combination (3 or 4 out of 4 must be None)
        from builtins import sum
        if sum(x is None for x in (dist, norm, inner, weight)) < 3:
            raise ValueError('invalid combination of options `weight`, '
                             '`dist`, `norm` and `inner`.')
        if weight is not None:
            if np.isscalar(weight):
                self._space_funcs = CudaFnConstWeighting(
                    weight, exponent=exponent)
            elif isinstance(weight, CudaFn.Vector):
                self._space_funcs = CudaFnVectorWeighting(
                    weight, exponent=exponent, copy_to_gpu=True)
            else:
                weight_ = np.asarray(weight)
                if weight_.ndim == 1:
                    self._space_funcs = CudaFnVectorWeighting(
                        weight_, exponent=exponent, copy_to_gpu=False)
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
            self._space_funcs = _CudaFnNoWeighting(exponent)

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of `x1` and `x2`, assigned to `out`.

        Calculate `z = a * x + b * y` using optimized CUDA routines.

        Parameters
        ----------
        a, b : `field` element
            Scalar to multiply `x` and `y` with.
        x, y : `CudaFn.Vector`
            The summands
        out : `CudaFn.Vector`
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
        x1, x2 : `CudaFn.Vector`

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
        x1, x2 : `CudaFn.Vector`
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
        """Calculate the norm of `x`.

        This method is implemented separately from `sqrt(inner(x,x))`
        for efficiency reasons.

        Parameters
        ----------
        x : `CudaFn.Vector`

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
        """The pointwise product of two vectors, assigned to `out`.

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
        CudaRn(3, 'float32').element([5.0, 6.0, 6.0])
        >>> out
        CudaRn(3, 'float32').element([5.0, 6.0, 6.0])
        """
        out.data.multiply(x1.data, x2.data)

    def _divide(self, x1, x2, out):
        """The pointwise division of two vectors, assigned to `out`.

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
        CudaRn(3, 'float32').element([5.0, 1.5, 1.0])
        >>> out
        CudaRn(3, 'float32').element([5.0, 1.5, 1.0])
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
        inner_str = '{}, {}'.format(self.size, dtype_repr(self.dtype))
        inner_str += _repr_space_funcs(self)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    class Vector(FnBase.Vector, CudaNtuples.Vector):

        """Representation of a `CudaFn` element.

        # TODO: document public interface
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            super().__init__(space, data)


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
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only real floating-point data types are allowed.

        kwargs : {'weight', 'exponent', 'dist', 'norm', 'inner'}
            See `CudaFn`
        """
        super().__init__(size, dtype, **kwargs)

        if not is_real_dtype(self._dtype):
            raise TypeError('data type {} not a real floating-point type.'
                            ''.format(dtype))

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


def _weighted(weight, attr, exponent, dist_using_inner=False):
    if np.isscalar(weight):
        weighting = CudaFnConstWeighting(
            weight, exponent)
    elif isinstance(weight, CudaFn.Vector):
        weighting = CudaFnVectorWeighting(
            weight, exponent=exponent, dist_using_inner=dist_using_inner,
            copy_to_gpu=True)
    else:
        weight_ = np.asarray(weight)
        if weight_.dtype == object:
            raise ValueError('bad weight {}'.format(weight))
        if weight_.ndim == 1:
            weighting = CudaFnVectorWeighting(
                weight_, exponent, dist_using_inner=dist_using_inner)
        elif weight_.ndim == 2:
            raise NotImplementedError('matrix weighting not implemented '
                                      'for CUDA spaces.')
#            weighting = CudaFnMatrixWeighting(
#                weight_, exponent, dist_using_inner=dist_using_inner)
        else:
            raise ValueError('array-like weight must have 1 or 2 dimensions, '
                             'but {} has {} dimensions.'
                             ''.format(weight, weight_.ndim))
    return getattr(weighting, attr)


def cu_weighted_inner(weight):
    """Weighted inner product on `CudaFn` spaces as free function.

    Parameters
    ----------
    weight : scalar, array-like or `CudaFn.Vector`
        Weight of the inner product. A scalar is interpreted as a
        constant weight and a 1-dim. array or a `CudaFn.Vector` as a
        weighting vector.

    Returns
    -------
    inner : callable
        Inner product function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    CudaFnConstWeighting, CudaFnVectorWeighting
    """
    return _weighted(weight, 'inner', exponent=2.0)


def cu_weighted_norm(weight, exponent=2.0):
    """Weighted norm on `CudaFn` spaces as free function.

    Parameters
    ----------
    weight : scalar, array-like or `CudaFn.Vector`
        Weight of the inner product. A scalar is interpreted as a
        constant weight and a 1-dim. array or a `CudaFn.Vector` as a
        weighting vector.
    exponent : positive float
        Exponent of the norm. If `weight` is a sparse matrix, only
        1.0, 2.0 and `inf` are allowed.

    Returns
    -------
    norm : callable
        Norm function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    CudaFnConstWeighting, CudaFnVectorWeighting
    """
    return _weighted(weight, 'norm', exponent=exponent)


def cu_weighted_dist(weight, exponent=2.0, use_inner=False):
    """Weighted distance on `CudaFn` spaces as free function.

    Parameters
    ----------
    weight : scalar, array-like or `CudaFn.Vector`
        Weight of the inner product. A scalar is interpreted as a
        constant weight and a 1-dim. array or a `CudaFn.Vector` as a
        weighting vector.
    exponent : positive float
        Exponent of the distance
    use_inner : bool, optional
        Calculate `dist(x, y)` as

        `sqrt(norm(x)**2 + norm(y)**2 - 2 * inner(x, y).real)`

        This avoids the creation of new arrays and is thus
        faster for large arrays. On the downside, it will not
        evaluate to exactly zero for equal (but not identical)
        `x` and `y`.

        Can only be used if `exponent` is 2.0.

    Returns
    -------
    dist : callable
        Distance function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    CudaFnConstWeighting, CudaFnVectorWeighting
    """
    return _weighted(weight, 'dist', exponent=exponent,
                     dist_using_inner=use_inner)


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


def _pnorm_default(x, p):
    if p == float('inf'):
        raise NotImplementedError('inf-norm not implemented.')
    elif int(p) != p:
        raise NotImplementedError('non-integer powers not implemented')
    # TODO: optimized version in C++ code?
    xp = abs(x)
    xp **= p
    return sum(xp)**(1/p)


def _pnorm_diagweight(x, p, w):
    if p == float('inf'):
        raise NotImplementedError('inf-norm not implemented.')
    elif int(p) != p:
        raise NotImplementedError('non-integer powers not implemented')
    # TODO: optimized version in C++ code?
    xp = abs(x)
    xp **= p
    xp *= w
    return sum(xp)**(1/p)


def _inner_default(x1, x2):
    return x1.data.inner(x2.data)


class _CudaFnWeighting(_FnWeightingBase):

    """Abstract base class for `CudaFn` weighting."""


class CudaFnVectorWeighting(_CudaFnWeighting):

    """Vector weighting for `CudaFn`.

    For exponent 2.0, a new weighted inner product with vector :math:`w`
    is defined as

    :math:`<a, b>_w := b^H (w * a)`

    with :math:`b^H` standing for transposed complex conjugate, and
    `w * a` being element-wise multiplication.

    For other exponents, only norm and dist are defined. In the case of
    exponent `inf`, the new norm is equal to the unweighted one,

    :math:`||a||_{w, \infty} := ||a||_\infty`,

    otherwise it is

    :math:`||a||_{w, p} := ||w^{1/p} * a||_p`.

    The vector may only have positive entries, otherwise it does not
    define an inner product or norm, respectively. This is not checked
    during initialization.
    """

    def __init__(self, vector, exponent=2.0, dist_using_inner=False,
                 copy_to_gpu=False):
        """Initialize a new instance.

        Parameters
        ----------
        vector : array-like, one-dim.
            Weighting vector of the inner product
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
            If `matrix` is a sparse matrix, only 1.0, 2.0 and `inf`
            are allowed.
        dist_using_inner : bool, optional
            Calculate `dist` using the formula

            norm(x-y)**2 = norm(x)**2 + norm(y)**2 - 2*inner(x, y).real

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) `x` and `y`.

            Can only be used if `exponent` is 2.0.
        copy_to_gpu : bool
            If `True`, the weights are stored as `CudaFn.Vector`,
            which consumes GPU memory but results in faster
            evaluation. If `False`, weights are stored as a NumPy
            array on the main memory.
        """
        # TODO: remove dist_using_inner in favor of a native dist
        # implementation
        super().__init__(exponent=exponent, dist_using_inner=dist_using_inner)
        if not isinstance(vector, CudaFn.Vector):
            self._vector = np.asarray(vector)
        else:
            self._vector = vector
        if self._vector.dtype == object:
            raise ValueError('invalid vector {}.'.format(vector))
        elif self._vector.ndim != 1:
            raise ValueError('vector {} is {}-dimensional instead of '
                             '1-dimensional.'
                             ''.format(vector, self._vector.ndim))
        if copy_to_gpu and not isinstance(self._vector, CudaFn.Vector):
            self._vector = CudaFn(self._vector.size,
                                  self._vector.dtype).element(self._vector)

    @property
    def vector(self):
        """Weighting vector of this inner product."""
        return self._vector

    def vector_is_valid(self):
        """Test if the vector is a valid weight, i.e. positive.

        Note
        ----
        This operation copies the vector to the CPU memory and uses
        `numpy.all`, which can be very time-consuming in total.
        """
        return np.all(self.vector > 0)

    def __eq__(self, other):
        """`inner.__eq__(other) <==> inner == other`.

        Returns
        -------
        equals : bool
            `True` if `other` is a `CudaFnVectorWeighting` instance
            with **identical** vector, `False` otherwise.

        See also
        --------
        equiv : test for equivalent inner products
        """
        if other is self:
            return True

        return (isinstance(other, CudaFnVectorWeighting) and
                self.vector is other.vector and
                self.exponent == other.exponent)

    def equiv(self, other):
        """Test if `other` is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            `True` if `other` is a `CudaFnWeighting` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is checked by entry-wise
            comparison of matrices/vectors/constant of this inner
            product and `other`.
        """
        # Optimization for equality
        if self == other:
            return True
        elif (not isinstance(other, _CudaFnWeighting) or
              self.exponent != other.exponent):
            return False
        elif isinstance(other, CudaFnConstWeighting):
            return np.array_equiv(self.vector, other.const)
        else:
            return False

    def inner(self, x1, x2):
        """Calculate the vector weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `Fn.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
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
        x : `Fn.Vector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the provided vector
        """
        if self.exponent == 2.0:
            return super().norm(x)
        elif self.exponent == float('inf'):
            return _pnorm_default(x, float('inf'))
        else:
            return float(_pnorm_diagweight(x, self.exponent, self.vector))

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{vector!r}'
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        if self._dist_using_inner:
            inner_fstr += ', dist_using_inner=True'

        inner_str = inner_fstr.format(vector=self.vector, ex=self.exponent)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__repr__() <==> repr(w)`."""
        if self.exponent == 2.0:
            return 'Weighting: vector =\n{}'.format(self.vector)
        else:
            return 'Weighting: p = {}, vector =\n{}'.format(self.exponent,
                                                            self.vector)


class CudaFnConstWeighting(_CudaFnWeighting):

    """Weighting of `CudaFn` by a constant.

    For exponent 2.0, a new weighted inner product with constant `c`
    is defined as

    :math:`<a, b>_c := c * b^H a`

    with :math:`b^H` standing for transposed complex conjugate, and
    `w * a` being element-wise multiplication.

    For other exponents, only norm and dist are defined. In the case of
    exponent `inf`, the new norm is equal to the unweighted one,

    :math:`||a||_{c, \infty} := ||a||_\infty`,

    otherwise it is

    :math:`||a||_{c, p} := c^{1/p} * ||a||_p`.

    The constant `c` must be positive.
    """

    def __init__(self, constant, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive float
            Weighting constant of the inner product.
        exponent : positive float
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super().__init__(exponent=exponent, dist_using_inner=False)
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
            `True` if `other` is a `CudaFnConstWeighting`
            instance with the same constant, `False` otherwise.
        """
        if other is self:
            return True
        else:
            return (isinstance(other, CudaFnConstWeighting) and
                    self.const == other.const and
                    self.exponent == other.exponent)

    def equiv(self, other):
        """Test if `other` is an equivalent weighting.

        Returns
        -------
        equivalent : bool
            `True` if `other` is a `_CudaFnWeighting` instance which
            yields the same result as this inner product for any
            input, `False` otherwise. This is the same as equality
            if `other` is a `CudaFnConstWeighting` instance, otherwise
            by entry-wise comparison of this inner product's constant
            with the matrix of `other`.
        """
        if other is self:
            return True
        elif isinstance(other, CudaFnConstWeighting):
            return self == other
        elif isinstance(other, _CudaFnWeighting):
            return other.equiv(self)
        else:
            return False

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `CudaFn.Vector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : float or complex
            The inner product of the two provided vectors
        """
        if self.exponent != 2.0:
            raise NotImplementedError('No inner product defined for '
                                      'exponent != 2 (got {}).'
                                      ''.format(self.exponent))
        else:
            return self.const * float(_inner_default(x1, x2))

    def norm(self, x):
        """Calculate the constant-weighted norm of a vector.

        Parameters
        ----------
        x1 : `CudaFn.Vector`
            Vector whose norm is calculated

        Returns
        -------
        norm : float
            The norm of the vector
        """
        if self.exponent == 2.0:
            from math import sqrt
            return sqrt(self.const) * float(_norm_default(x))
        elif self.exponent == float('inf'):  # Weighting irrelevant
            return float(_pnorm_default(x, float('inf')))
        else:
            return (self.const**(1/self.exponent) *
                    float(_pnorm_default(x, self.exponent)))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : `CudaFn.Vector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : float
            The distance between the vectors
        """
        if self.exponent == 2.0:
            from math import sqrt
            return sqrt(self.const) * float(_dist_default(x1, x2))
        elif self.exponent == float('inf'):
            # TODO: implement and optimize!
            return float(_pnorm_default(x1 - x2, float('inf')))
        else:
            # TODO: optimize!
            return (self.const**(1/self.exponent) *
                    _pnorm_default(x1 - x2, self.exponent))

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{}'
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        inner_str = inner_fstr.format(self.const, ex=self.exponent)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        if self.exponent == 2.0:
            return 'Weighting: const = {:.4}'.format(self.const)
        else:
            return 'Weighting: p = {}, const = {:.4}'.format(
                self.exponent, self.const)


class _CudaFnNoWeighting(CudaFnConstWeighting):

    """Weighting of `CudaFn` with constant 1.

    For exponent 2.0, the unweighted inner product is defined as

    :math:`<a, b> := b^H a`

    with :math:`b^H` standing for transposed complex conjugate.
    This is the CPU implementation using NumPy.

    For other exponents, only norm and dist are defined.
    """

    def __init__(self, exponent=2.0):
        """Initialize a new instance."""
        super().__init__(1.0, exponent=exponent)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = ''
        if self.exponent != 2.0:
            inner_fstr += ', exponent={ex}'
        inner_str = inner_fstr.format(ex=self.exponent).lstrip(', ')
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        if self.exponent == 2.0:
            return 'NoWeighting'
        else:
            return 'NoWeighting: p = {}'.format(self.exponent)


class _CudaFnCustomInnerProduct(_CudaFnWeighting):

    """Custom inner product on `CudaFn`."""

    def __init__(self, inner, dist_using_inner=True):
        """Initialize a new instance.

        Parameters
        ----------
        inner : callable
            The inner product implementation. It must accept two
            `CudaFn.Vector` arguments, return a complex number and
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
            `True` if `other` is an `CudaFnCustomInnerProduct`
            instance with the same inner product, `False` otherwise.
        """
        return (isinstance(other, _CudaFnCustomInnerProduct) and
                self.inner == other.inner)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        if self._dist_using_inner:
            inner_fstr += ',dist_using_inner={dist_u_i}'

        inner_str = inner_fstr.format(self.inner,
                                      dist_u_i=self._dist_using_inner)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__repr__()  # TODO: prettify?


class _CudaFnCustomNorm(_CudaFnWeighting):

    """Custom norm on `CudaFn`, removes `inner`."""

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : callable
            The norm implementation. It must accept an `CudaFn.Vector`
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
            `True` if `other` is an `CudaFnCustomNorm`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, _CudaFnCustomNorm) and
                self.norm == other.norm)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.norm)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__str__() <==> str(w)`."""
        return self.__repr__()  # TODO: prettify?


class _CudaFnCustomDist(_CudaFnWeighting):

    """Custom distance on `CudaFn`, removes `norm` and `inner`."""

    def __init__(self, dist):
        """Initialize a new instance.

        Parameters
        ----------
        dist : callable
            The distance function defining a metric on :math:`F^n`.
            It must accept two `CudaFn.Vector` arguments and fulfill the
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
            `True` if `other` is an `CudaFnCustomDist`
            instance with the same norm, `False` otherwise.
        """
        return (isinstance(other, _CudaFnCustomDist) and
                self.dist == other.dist)

    def __repr__(self):
        """`w.__repr__() <==> repr(w)`."""
        inner_fstr = '{!r}'
        inner_str = inner_fstr.format(self.dist)
        return '{}({})'.format(self.__class__.__name__, inner_str)

    def __str__(self):
        """`w.__repr__() <==> repr(w)`."""
        return self.__repr__()  # TODO: prettify?


try:
    CudaRn(1).element()
except (MemoryError, RuntimeError) as err:
    print(err)
    raise ImportError('Your GPU seems to be misconfigured. Skipping '
                      'CUDA-dependent modules.')


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
