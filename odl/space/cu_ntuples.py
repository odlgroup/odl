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

"""CUDA implementation of n-dimensional Cartesian spaces."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import int, super

import numpy as np

from odl.set.sets import RealNumbers
from odl.set.space import LinearSpaceVector
from odl.space.base_ntuples import (
    NtuplesBase, NtuplesBaseVector, FnBase, FnBaseVector)
from odl.space.weighting import (
    WeightingBase, VectorWeightingBase, ConstWeightingBase, NoWeightingBase,
    CustomInnerProductBase, CustomNormBase, CustomDistBase)
from odl.util.ufuncs import CudaNtuplesUFuncs
from odl.util.utility import dtype_repr

try:
    import odlpp.odlpp_cuda as cuda
    CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    CUDA_AVAILABLE = False


__all__ = ('CudaNtuples', 'CudaNtuplesVector',
           'CudaFn', 'CudaFnVector', 'CudaRn',
           'CUDA_DTYPES', 'CUDA_AVAILABLE',
           'CudaFnConstWeighting', 'CudaFnVectorWeighting',
           'cu_weighted_inner', 'cu_weighted_norm', 'cu_weighted_dist')


def _get_int_type():
    """Return the correct int vector type on the current platform."""
    if np.dtype(np.int).itemsize == 4:
        return 'CudaVectorInt32'
    elif np.dtype(np.int).itemsize == 8:
        return 'CudaVectorInt64'
    else:
        return 'CudaVectorIntNOT_AVAILABLE'


def _add_if_exists(dtype, name):
    """Add ``dtype`` to ``CUDA_DTYPES`` if it's available."""
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
CUDA_DTYPES = list(set(CUDA_DTYPES))  # Remove duplicates


class CudaNtuples(NtuplesBase):
    """The space `NtuplesBase`, implemented in CUDA."""

    def __init__(self, size, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        size : `int`
            The number entries per tuple
        dtype : `object`
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Check ``CUDA_DTYPES`` for a list of available data types.
        """

        if np.dtype(dtype) not in _TYPE_MAP_NPY2CUDA.keys():
            raise TypeError('data type {!r} not supported in CUDA'
                            ''.format(dtype))

        super().__init__(size, dtype)

        self._vector_impl = _TYPE_MAP_NPY2CUDA[self._dtype]

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : `array-like` or scalar, optional
            Input to initialize the new element.

            If ``inp`` is a `numpy.ndarray` of shape ``(size,)``
            and the same data type as this space, the array is wrapped,
            not copied.
            Other array-like objects are copied (with broadcasting
            if necessary).

            If a single value is given, it is copied to all entries.

            If both ``inp`` and ``data_ptr`` are `None`, an empty
            element is created with no guarantee of its state
            (memory allocation only).

        data_ptr : `int`, optional
            Memory address of a CUDA array container

            Cannot be combined with ``inp``.

        Returns
        -------
        element : `CudaNtuplesVector`
            The new element

        Notes
        -----
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
                return self.element_type(self, self._vector_impl(self.size))
            else:  # TODO: handle non-1 length strides
                return self.element_type(
                    self, self._vector_impl.from_pointer(data_ptr, self.size,
                                                         1))
        else:
            if data_ptr is None:
                if isinstance(inp, self._vector_impl):
                    return self.element_type(self, inp)
                elif isinstance(inp, self.element_type):
                    if inp in self:
                        return inp
                    else:
                        # Bulk-copy for non-matching dtypes
                        elem = self.element()
                        elem[:] = inp
                        return elem
                else:
                    # Array-like input. Need to go through a NumPy array
                    arr = np.array(inp, copy=False, dtype=self.dtype, ndmin=1)
                    if arr.shape != (self.size,):
                        raise ValueError('expected input shape {}, got {}.'
                                         ''.format((self.size,), arr.shape))
                    elem = self.element()
                    elem[:] = arr
                    return elem
            else:
                raise ValueError('cannot provide both inp and data_ptr.')

    @property
    def element_type(self):
        """ `CudaNtuplesVector` """
        return CudaNtuplesVector


class CudaNtuplesVector(NtuplesBaseVector, LinearSpaceVector):

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
        """Return ``self == other``.

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
        copy : `CudaNtuplesVector`
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
        return self.space.element_type(self.space, self.data.copy())

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
        out : `numpy.ndarray`
            Array in which the result should be written in-place.
            Has to be contiguous and of the correct dtype.

        Returns
        -------
        asarray : `numpy.ndarray`
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
        values : scalar or `CudaNtuplesVector`
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
        values : scalar, `array-like` or `CudaNtuplesVector`
            The value(s) that are to be assigned.

            If ``index`` is an `int`, ``value`` must be single value.

            If ``index`` is a `slice`, ``value`` must be broadcastable
            to the size of the slice (same size, shape (1,)
            or single value).

        Returns
        -------
        `None`

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
        if isinstance(values, CudaNtuplesVector):
            self.assign(values)  # use lincomb magic
        else:
            if isinstance(indices, slice):
                # Convert value to the correct type if needed
                value_array = np.asarray(values, dtype=self.space.dtype)

                if value_array.ndim == 0:
                    self.data.fill(values)
                else:
                    # Size checking is performed in c++
                    self.data.setslice(indices, value_array)
            else:
                self.data[int(indices)] = values

    @property
    def ufunc(self):
        """`CudaNtuplesUFuncs`, access to numpy style ufuncs.

        Examples
        --------
        >>> r2 = CudaRn(2)
        >>> x = r2.element([1, -2])
        >>> x.ufunc.absolute()
        CudaRn(2).element([1.0, 2.0])

        These functions can also be used with broadcasting

        >>> x.ufunc.add(3)
        CudaRn(2).element([4.0, 1.0])

        and non-space elements

        >>> x.ufunc.subtract([3, 3])
        CudaRn(2).element([-2.0, -5.0])

        There is also support for various reductions (sum, prod, min, max)

        >>> x.ufunc.sum()
        -1.0

        Also supports out parameter

        >>> y = r2.element([3, 4])
        >>> out = r2.element()
        >>> result = x.ufunc.add(y, out=out)
        >>> result
        CudaRn(2).element([4.0, 2.0])
        >>> result is out
        True

        Notes
        -----
        Not all ufuncs are currently optimized, some use the default numpy
        implementation. This can be improved in the future.

        See also
        --------
        odl.util.ufuncs.NtuplesBaseUFuncs
            Base class for ufuncs in `NtuplesBase` spaces.
        """
        return CudaNtuplesUFuncs(self)


class CudaFn(FnBase, CudaNtuples):

    """The space `FnBase`, implemented in CUDA.

    Requires the compiled ODL extension odlpp.
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
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Only scalar data types are allowed.

        weight : optional
            Use weighted inner product, norm, and dist. The following
            types are supported as ``weight``:

            `FnWeightingBase` :
            Use this weighting as-is. Compatibility with this
            space's elements is not checked during init.

            `float` :
            Weighting by a constant

            `array-like` :
            Weighting by a vector (1-dim. array, corresponds to
            a diagonal matrix). Note that the array is stored in
            main memory, which results in slower space functions
            due to a copy during evaluation.

            `CudaFnVector` :
            same as 1-dim. array-like, except that copying is
            avoided if the ``dtype`` of the vector is the
            same as this space's ``dtype``.

            Default: no weighting

            This option cannot be combined with ``dist``, ``norm``
            or ``inner``.

        exponent : positive `float`, optional
            Exponent of the norm. For values other than 2.0, no
            inner product is defined.

            This option is ignored if ``dist``, ``norm`` or
            ``inner`` is given.

            Default: 2.0

        dist : `callable`, optional
            The distance function defining a metric on the space.
            It must accept two `FnVector` arguments and
            fulfill the following mathematical conditions for any
            three vectors ``x, y, z``:

            - ``dist(x, y) >= 0``
            - ``dist(x, y) = 0``  if and only if  ``x = y``
            - ``dist(x, y) = dist(y, x)``
            - ``dist(x, y) <= dist(x, z) + dist(z, y)``

            This option cannot be combined with ``weight``,
            ``norm`` or ``inner``.

        norm : `callable`, optional
            The norm implementation. It must accept an
            `FnVector` argument, return a `float` and satisfy the
            following conditions for all vectors ``x, y`` and scalars
            ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``

            By default, ``norm(x)`` is calculated as ``inner(x, x)``.

            This option cannot be combined with ``weight``,
            ``dist`` or ``inner``.

        inner : `callable`, optional
            The inner product implementation. It must accept two
            `FnVector` arguments, return a element from
            the field of the space (real or complex number) and
            satisfy the following conditions for all vectors
            ``x, y, z`` and scalars ``s``:

            - ``<x, y> = conj(<y, x>)``
            - ``<s*x + y, z> = s * <x, z> + <y, z>``
            - ``<x, x> = 0``  if and only if  ``x = 0``

            This option cannot be combined with ``weight``,
            ``dist`` or ``norm``.
        """
        FnBase.__init__(self, size, dtype)
        CudaNtuples.__init__(self, size, dtype)

        dist = kwargs.pop('dist', None)
        norm = kwargs.pop('norm', None)
        inner = kwargs.pop('inner', None)
        weight = kwargs.pop('weight', None)
        exponent = kwargs.pop('exponent', 2.0)

        # Check validity of option combination (3 or 4 out of 4 must be None)
        if sum(x is None for x in (dist, norm, inner, weight)) < 3:
            raise ValueError('invalid combination of options `weight`, '
                             '`dist`, `norm` and `inner`.')
        if weight is not None:
            if isinstance(weight, WeightingBase):
                self._weighting = weight
            elif np.isscalar(weight):
                self._weighting = CudaFnConstWeighting(
                    weight, exponent=exponent)
            elif isinstance(weight, CudaFnVector):
                self._weighting = CudaFnVectorWeighting(
                    weight, exponent=exponent)
            else:
                # Must make a CudaFnVector from the array
                weight = self.element(np.asarray(weight))
                if weight.ndim == 1:
                    self._weighting = CudaFnVectorWeighting(
                        weight, exponent=exponent)
                else:
                    raise ValueError('invalid weight argument {!r}.'
                                     ''.format(weight))
        elif dist is not None:
            self._weighting = CudaFnCustomDist(dist)
        elif norm is not None:
            self._weighting = CudaFnCustomNorm(norm)
        elif inner is not None:
            # Use fast dist implementation
            self._weighting = CudaFnCustomInnerProduct(
                inner, dist_using_inner=True)
        else:  # all None -> no weighing
            self._weighting = CudaFnNoWeighting(exponent)

    @property
    def exponent(self):
        """Exponent of the norm and distance."""
        return self.weighting.exponent

    @property
    def weighting(self):
        """This space's weighting scheme."""
        return self._weighting

    @property
    def is_weighted(self):
        """Return `True` if the weighting is not `CudaFnNoWeighting`."""
        return not isinstance(self.weighting, CudaFnNoWeighting)

    @staticmethod
    def default_dtype(field):
        """Return the default of `CudaFn` data type for a given field.

        Parameters
        ----------
        field : `Field`
            Set of numbers to be represented by a data type.
            Currently supported: `RealNumbers`.

        Returns
        -------
        dtype : `type`
            Numpy data type specifier. The returned defaults are:

            ``RealNumbers()`` : , ``np.dtype('float32')``
        """
        if field == RealNumbers():
            return np.dtype('float32')
        else:
            raise ValueError('no default data type defined for field {}.'
                             ''.format(field))

    def _lincomb(self, a, x1, b, x2, out):
        """Linear combination of ``x1`` and ``x2``, assigned to ``out``.

        Calculate ``z = a * x + b * y`` using optimized CUDA routines.

        Parameters
        ----------
        a, b : `LinearSpace.field` `element`
            Scalar to multiply ``x`` and ``y`` with.
        x, y : `CudaFnVector`
            The summands
        out : `CudaFnVector`
            The Vector that the result is written to.

        Returns
        -------
        `None`

        Examples
        --------
        >>> r3 = CudaRn(3)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> out = r3.element()
        >>> r3.lincomb(2, x, 3, y, out)  # out is returned
        CudaRn(3).element([14.0, 19.0, 24.0])
        >>> out
        CudaRn(3).element([14.0, 19.0, 24.0])
        """
        out.data.lincomb(a, x1.data, b, x2.data)

    def _inner(self, x1, x2):
        """Calculate the inner product of x and y.

        Parameters
        ----------
        x1, x2 : `CudaFnVector`

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
        return self.weighting.inner(x1, x2)

    def _dist(self, x1, x2):
        """Calculate the distance between two vectors.

        Parameters
        ----------
        x1, x2 : `CudaFnVector`
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
        return self.weighting.dist(x1, x2)

    def _norm(self, x):
        """Calculate the norm of ``x``.

        This method is implemented separately from ``sqrt(inner(x,x))``
        for efficiency reasons.

        Parameters
        ----------
        x : `CudaFnVector`

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
        return self.weighting.norm(x)

    def _multiply(self, x1, x2, out):
        """The pointwise product of two vectors, assigned to ``out``.

        This is defined as:

        multiply(x, y, out) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

        Parameters
        ----------

        x1, x2 : `CudaFnVector`
            Factors in product
        out : `CudaFnVector`
            Element to which the result is written

        Returns
        -------
        `None`

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

        x1, x2 : `CudaFnVector`
            Factors in the product
        out : `CudaFnVector`
            Element to which the result is written

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
        return self.element_type(self, self._vector_impl(self.size, 0))

    def one(self):
        """Create a vector of ones."""
        return self.element_type(self, self._vector_impl(self.size, 1))

    def __eq__(self, other):
        """s.__eq__(other) <==> s == other.

        Returns
        -------
        equals : `bool`
            `True` if other is an instance of this space's type
            with the same ``size``, ``dtype`` and space functions,
            otherwise `False`.

        Examples
        --------
        >>> from numpy.linalg import norm
        >>> def dist(x, y, p):
        ...     return norm(x - y, ord=p)

        >>> from functools import partial
        >>> dist2 = partial(dist, p=2)
        >>> r3 = CudaRn(3, dist=dist2)
        >>> r3_same = CudaRn(3, dist=dist2)
        >>> r3  == r3_same
        True

        Different ``dist`` functions result in different spaces - the
        same applies for ``norm`` and ``inner``:

        >>> dist1 = partial(dist, p=1)
        >>> r3_1 = CudaRn(3, dist=dist1)
        >>> r3_2 = CudaRn(3, dist=dist2)
        >>> r3_1 == r3_2
        False

        Be careful with Lambdas - they result in non-identical function
        objects:

        >>> r3_lambda1 = CudaRn(3, dist=lambda x, y: norm(x-y, p=1))
        >>> r3_lambda2 = CudaRn(3, dist=lambda x, y: norm(x-y, p=1))
        >>> r3_lambda1 == r3_lambda2
        False
        """
        if other is self:
            return True

        return (CudaNtuples.__eq__(self, other) and
                self.weighting == other.weighting)

    def __repr__(self):
        """s.__repr__() <==> repr(s)."""
        if self.is_rn:
            class_name = 'CudaRn'
        elif self.is_cn:
            class_name = 'CudaCn'
        else:
            class_name = self.__class__.__name__

        inner_str = '{}'.format(self.size)
        if self.dtype != self.default_dtype(self.field):
            inner_str += ', {}'.format(dtype_repr(self.dtype))

        weight_str = self.weighting.repr_part
        if weight_str:
            inner_str += ', ' + weight_str
        return '{}({})'.format(class_name, inner_str)

    @property
    def element_type(self):
        """ `CudaFnVector` """
        return CudaFnVector


class CudaFnVector(FnBaseVector, CudaNtuplesVector):

    """Representation of a `CudaFn` element."""

    def __init__(self, space, data):
        """Initialize a new instance."""
        super().__init__(space, data)


def CudaRn(size, dtype='float32', **kwargs):

    """The real space :math:`R^n`, implemented in CUDA.

    Requires the compiled ODL extension odlpp.

    Parameters
    ----------
    size : positive `int`
        The number of dimensions of the space
    dtype : optional
        The data type of the storage array. Can be provided in any
        way the `numpy.dtype` function understands, most notably
        as built-in type, as one of NumPy's internal datatype
        objects or as string.

        Only real floating-point data types are allowed.

    kwargs : {'weight', 'exponent', 'dist', 'norm', 'inner'}
        See `CudaFn`
    """
    rn = CudaFn(size, dtype, **kwargs)

    if not rn.is_rn:
        raise TypeError('data type {!r} not a real floating-point type.'
                        ''.format(dtype))
    return rn


def _weighting(weight, exponent):
    """Return a weighting whose type is inferred from the arguments."""
    if np.isscalar(weight):
        weighting = CudaFnConstWeighting(
            weight, exponent)
    elif isinstance(weight, CudaFnVector):
        weighting = CudaFnVectorWeighting(
            weight, exponent=exponent)
    else:
        weight_ = np.asarray(weight)
        if weight_.dtype == object:
            raise ValueError('bad weight {}'.format(weight))
        if weight_.ndim == 1:
            weighting = CudaFnVectorWeighting(
                weight_, exponent)
        elif weight_.ndim == 2:
            raise NotImplementedError('matrix weighting not implemented '
                                      'for CUDA spaces.')
#            weighting = CudaFnMatrixWeighting(
#                weight_, exponent, dist_using_inner=dist_using_inner)
        else:
            raise ValueError('array-like weight must have 1 or 2 dimensions, '
                             'but {} has {} dimensions.'
                             ''.format(weight, weight_.ndim))
    return weighting


def cu_weighted_inner(weight):
    """Weighted inner product on `CudaFn` spaces as free function.

    Parameters
    ----------
    weight : scalar, `array-like` or `CudaFnVector`
        Weight of the inner product. A scalar is interpreted as a
        constant weight and a 1-dim. array or a `CudaFnVector`
        as a weighting vector.

    Returns
    -------
    inner : `callable`
        Inner product function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    CudaFnConstWeighting, CudaFnVectorWeighting
    """
    return _weighting(weight, exponent=2.0).inner


def cu_weighted_norm(weight, exponent=2.0):
    """Weighted norm on `CudaFn` spaces as free function.

    Parameters
    ----------
    weight : scalar, `array-like` or `CudaFnVector`
        Weight of the inner product. A scalar is interpreted as a
        constant weight and a 1-dim. array or a `CudaFnVector`
        as a weighting vector.
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
    CudaFnConstWeighting, CudaFnVectorWeighting
    """
    return _weighting(weight, exponent=exponent).norm


def cu_weighted_dist(weight, exponent=2.0):
    """Weighted distance on `CudaFn` spaces as free function.

    Parameters
    ----------
    weight : scalar, `array-like` or `CudaFnVector`
        Weight of the inner product. A scalar is interpreted as a
        constant weight and a 1-dim. array or a `CudaFnVector`
        as a weighting vector.
    exponent : positive `float`
        Exponent of the distance

    Returns
    -------
    dist : `callable`
        Distance function with given weight. Constant weightings
        are applicable to spaces of any size, for arrays the sizes
        of the weighting and the space must match.

    See also
    --------
    CudaFnConstWeighting, CudaFnVectorWeighting
    """
    return _weighting(weight, exponent=exponent).dist


def _dist_default(x1, x2):
    """Default Euclidean distance implementation."""
    return x1.data.dist(x2.data)


def _pdist_default(x1, x2, p):
    """Default p-distance implementation."""
    if p == float('inf'):
        raise NotImplementedError('inf-norm not implemented.')
    return x1.data.dist_power(x2.data, p)


def _pdist_diagweight(x1, x2, p, w):
    """Diagonally weighted p-distance implementation."""
    return x1.data.dist_weight(x2.data, p, w.data)


def _norm_default(x):
    """Default Euclidean norm implementation."""
    return x.data.norm()


def _pnorm_default(x, p):
    """Default p-norm implementation."""
    if p == float('inf'):
        raise NotImplementedError('inf-norm not implemented.')
    return x.data.norm_power(p)


def _pnorm_diagweight(x, p, w):
    """Diagonally weighted p-norm implementation."""
    if p == float('inf'):
        raise NotImplementedError('inf-norm not implemented.')
    return x.data.norm_weight(p, w.data)


def _inner_default(x1, x2):
    """Default Euclidean inner product implementation."""
    return x1.data.inner(x2.data)


def _inner_diagweight(x1, x2, w):
    return x1.data.inner_weight(x2.data, w.data)


class CudaFnVectorWeighting(VectorWeightingBase):

    """Vector weighting for `CudaFn`.

    For exponent 2.0, a new weighted inner product with vector ``w``
    is defined as::

        <a, b>_w := <w * a, b> = b^H (w * a)

    with ``b^H`` standing for transposed complex conjugate and
    ``w * a`` for element-wise multiplication.

    For other exponents, only norm and dist are defined. In the case of
    exponent ``inf``, the weighted norm is

        ||a||_{w, inf} := ||w * a||_inf

    otherwise it is::

        ||a||_{w, p} := ||w^{1/p} * a||

    Note that this definition does **not** fulfill the limit property
    in ``p``, i.e.::

        ||x||_{w, p} --/-> ||x||_{w, inf}  for p --> inf

    unless ``w = (1,...,1)``.

    The vector may only have positive entries, otherwise it does not
    define an inner product or norm, respectively. This is not checked
    during initialization.
    """

    def __init__(self, vector, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        vector : `CudaFnVector`
            Weighting vector of the inner product, norm and distance
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        if not isinstance(vector, CudaFnVector):
            raise TypeError('vector {!r} is not a CudaFnVector instance.'
                            ''.format(vector))

        super().__init__(vector, impl='cuda', exponent=exponent,
                         dist_using_inner=False)

    def inner(self, x1, x2):
        """Calculate the vector weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `CudaFnVector`
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
            return _inner_diagweight(x1, x2, self.vector)

    def norm(self, x):
        """Calculate the vector-weighted norm of a vector.

        Parameters
        ----------
        x : `CudaFnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : `float`
            The norm of the provided vector
        """
        if self.exponent == float('inf'):
            raise NotImplementedError('inf norm not implemented yet.')
        else:
            return _pnorm_diagweight(x, self.exponent, self.vector)

    def dist(self, x1, x2):
        """Calculate the vector-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : `CudaFnVector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
            The distance between the vectors
        """
        if self.exponent == float('inf'):
            raise NotImplementedError('inf norm not implemented yet.')
        else:
            return _pdist_diagweight(x1, x2, self.exponent, self.vector)


class CudaFnConstWeighting(ConstWeightingBase):

    """Weighting of `CudaFn` by a constant.

    For exponent 2.0, a new weighted inner product with constant
    ``c`` is defined as::

        <a, b>_c = c * <a, b> = c * b^H a

    with ``b^H`` standing for transposed complex conjugate.

    For other exponents, only norm and dist are defined. In the case of
    exponent ``inf``, the weighted norm is defined as::

        ||a||_{c, inf} := c ||a||_inf

    otherwise it is::

        ||a||_{c, p} := c^{1/p}  ||a||_p

    Note that this definition does **not** fulfill the limit property
    in ``p``, i.e.::

        ||a||_{c,p} --/-> ||a||_{c,inf}  for p --> inf

    unless ``c = 1``.

    The constant must be positive, otherwise it does not define an
    inner product or norm, respectively.
    """

    def __init__(self, constant, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        constant : positive finite `float`
            Weighting constant of the inner product.
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super().__init__(constant, impl='cuda', exponent=exponent,
                         dist_using_inner=False)

    def inner(self, x1, x2):
        """Calculate the constant-weighted inner product of two vectors.

        Parameters
        ----------
        x1, x2 : `CudaFnVector`
            Vectors whose inner product is calculated

        Returns
        -------
        inner : `float` or `complex`
            The inner product of the two vectors
        """
        if self.exponent != 2.0:
            raise NotImplementedError('No inner product defined for '
                                      'exponent != 2 (got {}).'
                                      ''.format(self.exponent))
        else:
            return self.const * _inner_default(x1, x2)

    def norm(self, x):
        """Calculate the constant-weighted norm of a vector.

        Parameters
        ----------
        x1 : `CudaFnVector`
            Vector whose norm is calculated

        Returns
        -------
        norm : `float`
            The norm of the vector
        """
        if self.exponent == float('inf'):
            raise NotImplementedError
            # Example impl
            # return self.const * float(_pnorm_default(x, self.exponent))
        else:
            return (self.const ** (1 / self.exponent) *
                    float(_pnorm_default(x, self.exponent)))

    def dist(self, x1, x2):
        """Calculate the constant-weighted distance between two vectors.

        Parameters
        ----------
        x1, x2 : `CudaFnVector`
            Vectors whose mutual distance is calculated

        Returns
        -------
        dist : `float`
            The distance between the vectors
        """
        if self.exponent == float('inf'):
            raise NotImplementedError
        else:
            return (self.const ** (1 / self.exponent) *
                    _pdist_default(x1, x2, self.exponent))


class CudaFnNoWeighting(NoWeightingBase, CudaFnConstWeighting):

    """Weighting of `CudaFn` with constant 1.

    For exponent 2.0, the unweighted inner product is defined as::

        <a, b> := b^H a

    with ``b^H`` standing for transposed complex conjugate.

    For other exponents, only norm and dist are defined.
    """

    # Implement singleton pattern for efficiency in the default case
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern if ``exponent==2.0``."""
        if len(args) == 0:
            exponent = kwargs.pop('exponent', 2.0)
        else:
            exponent = args[0]
            args = args[1:]

        if exponent == 2.0:
            if not cls._instance:
                cls._instance = super().__new__(cls, *args, **kwargs)
            return cls._instance
        else:
            return super().__new__(cls, *args, **kwargs)

    def __init__(self, exponent=2.0):
        """Initialize a new instance.

        Parameters
        ----------
        exponent : positive `float`
            Exponent of the norm. For values other than 2.0, the inner
            product is not defined.
        """
        super().__init__(exponent=exponent, impl='cuda',
                         dist_using_inner=False)


class CudaFnCustomInnerProduct(CustomInnerProductBase):

    """Class for handling a user-specified inner product on `CudaFn`."""

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
            Calculate ``dist`` using the following formula::

                ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * Re <x, y>

            This avoids the creation of new arrays and is thus faster
            for large arrays. On the downside, it will not evaluate to
            exactly zero for equal (but not identical) ``x`` and ``y``.
        """
        super().__init__(inner, impl='cuda',
                         dist_using_inner=dist_using_inner)


class CudaFnCustomNorm(CustomNormBase):

    """Class for handling a user-specified norm in `CudaFn`.

    Note that this removes ``inner``.
    """

    def __init__(self, norm):
        """Initialize a new instance.

        Parameters
        ----------
        norm : `callable`
            The norm implementation. It must accept a `CudaFnVector`
            argument, return a `float` and satisfy the following
            conditions for all vectors ``x, y`` and scalars ``s``:

            - ``||x|| >= 0``
            - ``||x|| = 0``  if and only if  ``x = 0``
            - ``||s * x|| = |s| * ||x||``
            - ``||x + y|| <= ||x|| + ||y||``
        """
        super().__init__(norm, impl='cuda')


class CudaFnCustomDist(CustomDistBase):

    """Class for handling a user-specified distance in `CudaFn`.

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
        super().__init__(dist, impl='cuda')


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
