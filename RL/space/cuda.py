# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.

""" Module for spaces whose elements are in R^n

This is the default implementation of R^n where the
data is stored on a GPU.
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import str, super
from future import standard_library

# External module imports
import numpy as np
from numbers import Integral

# RL imports
import RL.space.space as spaces
import RL.space.set as sets
import RLcpp.PyCuda
from RL.utility.utility import errfmt

standard_library.install_aliases()


class CudaRn(spaces.HilbertSpace, spaces.Algebra):
    """The real space R^n, implemented in CUDA

    Requires the compiled RL extension RLcpp.

    Parameters
    ----------

    n : int
        The dimension of the space
    """

    def __init__(self, dim):
        if not isinstance(dim, Integral) or dim < 1:
            raise TypeError(errfmt('''
            'dim' ({}) has to be a positive integer'''.format(np)))
        self._dim = dim
        self._field = sets.RealNumbers()
        self.impl = RLcpp.PyCuda.CudaRNImpl(dim)

    def element(self, data=None, **kwargs):
        """ Returns a vector of zeros

        CUDA memory is always initialized
        TODO: rewrite

        Parameters
        ----------
        None

        Returns
        -------
        CudaRn.Vector instance


        Examples
        --------

        >>> rn = CudaRn(3)
        >>> y = rn.element()
        >>> y in rn
        True
        >>> y.assign(rn.zero())
        >>> y
        CudaRn(3).element([ 0.,  0.,  0.])

        Creates an element in CudaRn

        Parameters
        ----------
        The method has two call patterns, the first is:

        *args : numpy.ndarray
                Array that will be copied to the GPU.
                Data is not modified or bound.
                The shape of the array must be (n,)

        **kwargs : None

        The second pattern is to create a new numpy array which will then
        be copied to the GPU. In this case

        *args : Options for numpy.array constructor
        **kwargs : Options for numpy.array constructor

        Returns
        -------
        CudaRn.Vector instance


        Examples
        --------

        >>> rn = CudaRn(3)
        >>> x = rn.element(np.array([1, 2, 3]))
        >>> x
        CudaRn(3).element([ 1.,  2.,  3.])
        >>> y = rn.element([1, 2, 3])
        >>> y
        CudaRn(3).element([ 1.,  2.,  3.])

        """

        if isinstance(data, RLcpp.PyCuda.CudaRNVectorImpl):
            return self.Vector(self, data)
        elif data is None:
            return self.element(self.impl.empty())
        elif isinstance(data, np.ndarray):  # Create from numpy array
            if data.shape != (self.dim,):
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape shape {}
                '''.format(data, data.shape, (self.dim,))))

            data = data.astype(np.float64, copy=False)

            # Create result and assign (could be optimized to one call)
            elem = self.element()
            elem[:] = data
            return elem
        else:  # Create from intermediate numpy array
            as_array = np.array(data, dtype=np.float64, **kwargs)
            return self.element(as_array)

    def _inner(self, x, y):
        """ Calculates the inner product of x and y

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

        Also has member inner
        >>> x.inner(y)
        20.0
        """

        return self.impl.inner(x.data, y.data)

    def _norm(self, x):
        """ Calculates the 2-norm of x

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

        Also has member inner
        >>> x.norm()
        7.0
        """

        return self.impl.norm(x.data)

    def _lincomb(self, z, a, x, b, y):
        """ Linear combination of x and y

        z = a*x + b*y

        Parameters
        ----------
        z : CudaRn.Vector
            The Vector that the result should be written to.
        a : RealNumber
            Scalar to multiply `x` with.
        x : CudaRn.Vector
            The first of the summands
        b : RealNumber
            Scalar to multiply `y` with.
        y : CudaRn.Vector
            The second of the summands

        Returns
        -------
        None

        Examples
        --------
        >>> rn = CudaRn(3)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element([4, 5, 6])
        >>> z = rn.element()
        >>> rn.lincomb(z, 2, x, 3, y)
        >>> z
        CudaRn(3).element([ 14.,  19.,  24.])
        """

        self.impl.linComb(z.data, a, x.data, b, y.data)

    def _multiply(self, x, y):
        """ Calculates the pointwise product of two vectors and assigns the
        result to `y`

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
        CudaRn(3).element([ 5.,  6.,  6.])
        """
        self.impl.multiply(x.data, y.data)

    def zero(self):
        """ Returns a vector of zeros

        Parameters
        ----------
        None

        Returns
        -------
        CudaRn.Vector instance with all elements set to zero (0.0)


        Examples
        --------

        >>> rn = CudaRn(3)
        >>> y = rn.zero()
        >>> y
        CudaRn(3).element([ 0.,  0.,  0.])
        """
        return self.element(self.impl.zero())

    @property
    def field(self):
        """ The underlying field of R^n is the set of real numbers

        Parameters
        ----------
        None

        Returns
        -------
        RealNumbers instance


        Examples
        --------

        >>> rn = CudaRn(3)
        >>> rn.field
        RealNumbers()
        """
        return self._field

    @property
    def dim(self):
        """ The dimension of this space

        Parameters
        ----------
        None

        Returns
        -------
        Integer


        Examples
        --------

        >>> rn = CudaRn(3)
        >>> rn.dim
        3
        """
        return self._dim

    def equals(self, other):
        """ Verifies that other is a CudaRn instance of dimension `n`

        Parameters
        ----------
        other : any object
                The object to check for equality

        Returns
        -------
        boolean      True if equal, else false

        Examples
        --------

        >>> r3 = CudaRn(3)
        >>> r3.equals(r3)
        True

        >>> r3a, r3b = CudaRn(3), CudaRn(3)
        >>> r3a.equals(r3b)
        True

        >>> r3, r4 = CudaRn(3), CudaRn(4)
        >>> r3.equals(r4)
        False

        >>> r3, r4 = CudaRn(3), CudaRn(4)
        >>> r3 == r3
        True
        >>> r3 == r4
        False
        >>> r3 != r4
        True
        """
        return isinstance(other, CudaRn) and self.dim == other.dim

    def __str__(self):
        return "CudaRn(" + str(self.dim) + ")"

    def __repr__(self):
        return "CudaRn(" + str(self.dim) + ")"

    class Vector(spaces.HilbertSpace.Vector, spaces.Algebra.Vector):
        """ A R^n vector represented in CUDA

        Parameters
        ----------

        space : CudaRn
                Instance of CudaRn this vector lives in
        data : RLcpp.PyCuda.CudaRNVectorImpl
                    Underlying data-representation to be used by this vector
        """
        def __init__(self, space, data):
            super().__init__(space)
            if not isinstance(data, RLcpp.PyCuda.CudaRNVectorImpl):
                raise TypeError(errfmt('''
                'data' ({}) must be a CudaRNVectorImpl instance
                '''.format(data)))
            self._data = data

        @property
        def data(self):
            """ Get the data of this Vector

            Parameters
            ----------
            None

            Returns
            -------
            ptr : RLcpp.PyCuda.CudaRNVectorImpl
                  Underlying cuda data representation
            """
            return self._data

        @property
        def data_ptr(self):
            """ Get a raw pointer to the data of this Vector

            Parameters
            ----------
            None

            Returns
            -------
            ptr : Int
                  Pointer to the CUDA data of this vector
            """
            return self._data.dataPtr()

        def __str__(self):
            return str(self[:])

        def __repr__(self):
            """ Get a representation of this vector

            Parameters
            ----------
            None

            Returns
            -------
            repr : string
                   String representation of this vector

            Examples
            --------

            >>> rn = CudaRn(3)
            >>> x = rn.element([1, 2, 3])
            >>> y = eval(repr(x))
            >>> y
            CudaRn(3).element([ 1.,  2.,  3.])
            """
            val_str = repr(self[:]).lstrip('array(').rstrip(')')
            return repr(self.space) + '.element(' + val_str + ')'

        def __len__(self):
            """ Get the dimension of the underlying space
            """
            return self.space.n

        def __getitem__(self, index):
            """ Access values of this vector.

            This will cause the values to be copied to CPU
            which is a slow operation.

            Parameters
            ----------

            index : int or slice
                    The position(s) that should be accessed

            Returns
            -------
            If index is an `int`
            float, value at index

            If index is an `slice`
            numpy.ndarray instance with the values at the slice


            Examples
            --------

            >>> rn = CudaRn(3)
            >>> y = rn.element([1, 2, 3])
            >>> y[0]
            1.0
            >>> y[1:2]
            array([ 2.])

            """
            if isinstance(index, slice):
                return self.data.getSlice(index)
            else:
                return self.data.__getitem__(index)

        def __setitem__(self, index, value):
            """ Set values of this vector

            This will cause the values to be copied to CPU
            which is a slow operation.

            Parameters
            ----------

            index : int or slice
                    The position(s) that should be set
            value : Real or Array-Like
                    The values that should be assigned.
                    If index is an integer,
                    value should be a Number convertible to float.
                    If index is a slice,
                    value should be an Array-Like of the same
                    size as the slice.

            Returns
            -------
            None


            Examples
            --------


            >>> rn = CudaRn(3)
            >>> y = rn.element([1, 2, 3])
            >>> y[0] = 5
            >>> y
            CudaRn(3).element([ 5.,  2.,  3.])
            >>> y[1:3] = [7, 8]
            >>> y
            CudaRn(3).element([ 5.,  7.,  8.])
            >>> y[:] = np.array([0, 0, 0])
            >>> y
            CudaRn(3).element([ 0.,  0.,  0.])

            """

            if isinstance(index, slice):
                # Convert value to the correct type
                if not isinstance(value, np.ndarray):
                    value = np.array(value, dtype=np.float64)

                value = value.astype(np.float64, copy=False)

                self.data.setSlice(index, value)
            else:
                self.data.__setitem__(index, value)


# Methods, TODO: move
def abs(inp, outp):
    RLcpp.PyCuda.abs(inp.data, outp.data)

def sign(inp, outp):
    RLcpp.PyCuda.sign(inp.data, outp.data)

def add_scalar(inp, scal, outp):
    RLcpp.PyCuda.addScalar(inp.data, scal, outp.data)

def max_vector_scalar(inp, scal, outp):
    RLcpp.PyCuda.maxVectorScalar(inp.data, scal, outp.data)

def max_vector_vector(inp1, inp2, outp):
    RLcpp.PyCuda.maxVectorVector(inp1.data, inp2.data, outp.data)

def sum(inp):
    return RLcpp.PyCuda.sum(inp.data)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
