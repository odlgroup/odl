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

"""CUDA implementation of n-dimensional Cartesian spaces.

# TODO: document public interface
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import str, super
from future import standard_library

# External module imports
import numpy as np

# ODL imports
from odl.space.cartesian import NtuplesBase, FnBase
import odlpp.odlpp_cuda as cuda
from odlpp.odlpp_cuda import CudaVectorUchar, CudaVectorFloat

standard_library.install_aliases()


_type_map_npy2cuda = {np.dtype('float32'): CudaVectorFloat,
                      np.dtype('uint8'): CudaVectorUchar}


class CudaNtuples(NtuplesBase):

    """The set of `n`-tuples of arbitrary type, implemented in CUDA.

    See also
    --------
    See the module documentation for attributes, methods etc.
    """

    def __init__(self, dim, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        dim : `Integral`
            The number entries per tuple
        dtype : `object`
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.

            Currently supported: 'float32', 'uint8'
        """
        super().__init__(dim, dtype)
        if self._dtype not in _type_map_npy2cuda.keys():
            raise TypeError('data type {} not supported in '.format(dtype))

        self._vector_impl = _type_map_npy2cuda[self._dtype]

    def element(self, inp=None, data_ptr=None):
        """Create a new element.

        Parameters
        ----------
        inp : array-like or scalar, optional
            Input to initialize the new element.

            If `inp` is a `numpy.ndarray` of shape `(dim,)` and the
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

        >>> y = uc3.element([1, 2, 3])
        >>> y

        """
        if inp is None and data_ptr is None:
            return self.Vector(self, self._vector_impl(self.dim))
        elif inp is None:
            return self.Vector(
                self, self._vector_impl.from_pointer(data_ptr, self.dim))
        elif data_ptr is None:
            # TODO: scalar assignment will fail. Implement a fill()
            # method for that case
            elem = self.element()
            elem[:] = inp
            return elem
        else:
            raise TypeError("Cannot provide both inp and data_ptr")

    class Vector(NtuplesBase.Vector):

        """Representation of a `CudaNtuples` element.

        See also
        --------
        See the module documentation for attributes, methods etc.
        """

        def __init__(self, space, data):
            """Initialize a new instance."""
            if not isinstance(space, CudaNtuples):
                raise TypeError('{!r} not a `CudaNtuples` instance.'
                                ''.format(space))

            super().__init__(space, data)

            if not isinstance(data, self._space._vector_impl):
                raise TypeError('data {!r} not a `{}` instance.'
                                ''.format(data, self._space._vector_impl))

        @property
        def data_ptr(self):
            """A raw pointer to the data of this vector."""
            return self._data.data_ptr()

        @property
        def itemsize(self):
            """The size in bytes of the data type."""
            # TODO: Currently hardcoded to float, change this
            return 4

        def equals(self, other):
            """Test if `other` is equal to this vector."""
            raise NotImplementedError

        def copy(self):
            """Create an identical (deep) copy of this vector."""
            # TODO: implement
            raise NotImplementedError

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
            >>> y[1:2]
            array([2], dtype=uint8)
            """
            try:
                return self.data.__getitem__(int(indices))
            except TypeError:
                return self.data.getslice(indices)

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

            >>> y[1:3] = [7, 8]
            >>> y

            >>> y[:] = np.array([0, 0, 0])
            >>> y

            """
            if isinstance(values, CudaNtuples.Vector):
                raise NotImplementedError
                # TODO: implement
                # return self.data.__setitem__(indices, values.data)
            else:
                # Convert value to the correct type if needed
                values = np.asarray(values, dtype=self.space._dtype)
                try:
                    return self.data.__setitem__(int(indices), values[0])
                except TypeError:
                    # Size checking is performed in c++
                    return self.data.setslice(indices, values)


class CudaFn(FnBase, CudaNtuples):

    """The space F^n, implemented in CUDA.

    Requires the compiled ODL extension odlpp.

    # TODO: document public interface
    """

    def __init__(self, dim, dtype):
        """Initialize a new instance.

        Parameters
        ----------
        dim : `Integral`
            The number entries per tuple
        dtype : `object`
            The data type for each tuple entry. Can be provided in any
            way the `numpy.dtype()` function understands, most notably
            as built-in type, as one of NumPy's internal datatype
            objects or as string.
            Only scalar data types (numbers) are allowed.

            Currently supported: 'float32', 'uint8'
        """
        super().__init__(dim, dtype)
        CudaNtuples.__init__(self, dim, dtype)

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

        """
        z.data.linComb(a, x.data, b, y.data)

    def _inner(self, x, y):
        """Calculate the inner product of x and y.

        Parameters
        ----------
        x : CudaRn.Vector
        y : CudaRn.Vector

        Returns
        -------
        inner: float
            The inner product of x and y


        Examples
        --------

        >>> rn = CudaRn(3)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element([3, 1, 5])
        >>> rn.inner(x, y)
        20.0
        """
        return x.data.inner(y.data)

    def _norm(self, x):
        """Calculate the 2-norm of x.

        This method is implemented separately from `sqrt(inner(x,x))`
        for efficiency reasons.

        Parameters
        ----------
        x : CudaRn.Vector

        Returns
        -------
        norm : float
            The 2-norm of x


        Examples
        --------

        >>> rn = CudaRn(3)
        >>> x = rn.element([2, 3, 6])
        >>> rn.norm(x)
        7.0
        """
        return x.data.norm()

    def _multiply(self, x, y):
        """The pointwise product of two vectors, assigned to `y`.

        This is defined as:

        multiply(x, y) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

        Parameters
        ----------

        x : CudaRn.Vector
            read from
        y : CudaRn.Vector
            read from and written to

        Returns
        -------
        None

        Examples
        --------

        >>> rn = CudaRn(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> rn.multiply(x, y)
        >>> y
        CudaRn(3).element([5.0, 6.0, 6.0])
        """
        y.data.multiply(x.data)

    def zero(self):
        """Create a vector of zeros."""
        return self.Vector(self, self._vector_impl(self.dim, 0))

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

    def __init__(self, dim, dtype=np.float32):
        """Initialize a new instance.

        Only real floating-point types are allowed.
        """
        super().__init__(dim, dtype)

        if not np.isrealobj(np.empty(0, dtype=self._dtype)):
            raise TypeError('data type {} not a real floating-point type.'
                            ''.format(dtype))

    class Vector(CudaFn.Vector):
        pass


# Methods
# TODO: move
def abs(inp, outp):
    cuda.abs(inp.data, outp.data)


def sign(inp, outp):
    cuda.sign(inp.data, outp.data)


def add_scalar(inp, scal, outp):
    cuda.add_scalar(inp.data, scal, outp.data)


def max_vector_scalar(inp, scal, outp):
    cuda.max_vector_scalar(inp.data, scal, outp.data)


def max_vector_vector(inp1, inp2, outp):
    cuda.max_vector_vector(inp1.data, inp2.data, outp.data)


def sum(inp):
    return cuda.sum(inp.data)


try:
    CudaRn(1).element()
except MemoryError:
    print('Warning: Your GPU seems to be misconfigured. Skipping '
          'CUDA-dependent modules.')
    raise ImportError


if __name__ == '__main__':
    from doctest import testmod, NORMALIZE_WHITESPACE
    testmod(optionflags=NORMALIZE_WHITESPACE)
