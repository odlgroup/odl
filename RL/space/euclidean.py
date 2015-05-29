""" Module for spaces whose elements are in R^n

This is the default implementation of R^n and the
corresponding NormedRN and EuclideanSpace.

The underlying datarepresentation used is Numpy Arrays.
"""
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


# Imports for common Python 2/3 codebase
from __future__ import unicode_literals, print_function, division
from __future__ import absolute_import
from builtins import str, super
from future import standard_library

# External module imports
import numpy as np
from scipy.lib.blas import get_blas_funcs
from numbers import Integral

# RL imports
from RL.space.space import LinearSpace, Algebra, HilbertSpace, NormedSpace
from RL.space.set import RealNumbers
from RL.utility.utility import errfmt

standard_library.install_aliases()


class RN(LinearSpace):
    """The real space R^n

    Parameters
    ----------

    n : int
        The dimension of the space
    """

    def __init__(self, n):
        if not isinstance(n, Integral) or n < 1:
            raise TypeError('n ({}) has to be a positive integer'.format(np))
        self._dim = n
        self._field = RealNumbers()
        self._axpy, self._scal, self._copy = get_blas_funcs(['axpy',
                                                             'scal',
                                                             'copy'])

    def element(self, data=None, **kwargs):
        """ Returns an arbitrary vector

        more efficient than zeros.
        TODO: rewrite

        Parameters
        ----------
        None

        Returns
        -------
        RN.Vector instance

        Note
        ----
        The values of the returned vector may be _anything_ including
        inf and NaN. Thus operations such as empty() * 0 need not return
        the zero vector.

        Examples
        --------

        >>> rn = RN(3)
        >>> x = rn.element()
        >>> x in rn
        True

        Creates an element in RN

        Parameters
        ----------
        The method has two call patterns, the first is:

        *args : numpy.ndarray
                Array that will be used as the underlying representation
                The dtype of the array must be float64
                The shape of the array must be (n,)

        **kwargs : None

        The second pattern is to create a new numpy array, in this case

        *args : Options for numpy.array constructor
        **kwargs : Options for numpy.array constructor

        Returns
        -------
        RN.Vector instance


        Examples
        --------

        >>> rn = RN(3)
        >>> x = rn.element(np.array([1., 2., 3.]))
        >>> x
        RN(3).element([ 1.,  2.,  3.])
        >>> y = rn.element([1, 2, 3])
        >>> y
        RN(3).element([ 1.,  2.,  3.])

        """

        dtype = kwargs.pop('dtype', np.float64)

        if data is None:
            data = np.empty(self._dim, dtype=dtype, **kwargs)

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype, **kwargs)
        else:
            if data.shape != (self._dim,):
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape shape {}
                '''.format(data, data.shape, (self.dim,))))

            if data.dtype != np.float64:
                raise ValueError(errfmt('''
                Input numpy array ({}) is of type {}, expected float64
                '''.format(data, data.dtype)))

        return self.Vector(self, data)

    def _lincomb(self, z, a, x, b, y):
        """ Implement y = a*x + b*y using optimized BLAS rutines

        Parameters
        ----------
        z : RNVector
            The Vector that the result should be written to.
        a : RealNumber
            Scalar to multiply `x` with.
        x : RNVector
            The first of the summands
        b : RealNumber
            Scalar to multiply `y` with.
        y : RNVector
            The second of the summands

        Returns
        -------
        None

        Examples
        --------
        >>> rn = RN(3)
        >>> x = rn.element([1, 2, 3])
        >>> y = rn.element([4, 5, 6])
        >>> z = rn.element()
        >>> rn.lincomb(z, 2, x, 3, y)
        >>> z
        RN(3).element([ 14.,  19.,  24.])

        """

        if x is y and b != 0:
            # If x is aligned with y, we are looking at:     z = (a+b)*x
            self._lincomb(z, a+b, x, 0, x)
        elif z is x and z is y:
            # If all the vectors are aligned we have:        z = (a+b)*z
            self._scal(a+b, z.data)
        elif z is x:
            # If z is aligned with x we have                 z = a*z + b*y
            if a != 1:
                self._scal(a, z.data)
            if b != 0:
                self._axpy(y.data, z.data, self._dim, b)
        elif z is y:
            # If z is aligned with y we have                 z = a*x + b*z
            if b != 1:
                self._scal(b, z.data)
            if a != 0:
                self._axpy(x.data, z.data, self._dim, a)
        else:
            # We have exhausted all alignment options, so x != y != z
            # We now optimize for various values of a and b
            if b == 0:
                if a == 0:  # Zero assignment                z = 0
                    z.data[:] = 0
                else:                                       # z = a*x
                    self._copy(x.data, z.data)
                    if a != 1:
                        self._scal(a, z.data)
            else:
                if a == 0:                                  # z = b*y
                    self._copy(y.data, z.data)
                    if b != 1:
                        self._scal(b, z.data)

                elif a == 1:                                # z = x + b*y
                    self._copy(x.data, z.data)
                    self._axpy(y.data, z.data, self._dim, b)
                else:                                       # z = a*x + b*y
                    self._copy(y.data, z.data)
                    if b != 1:
                        self._scal(b, z.data)
                    self._axpy(x.data, z.data, self._dim, a)

    def zero(self):
        """ Returns a vector of zeros

        Parameters
        ----------
        None

        Returns
        -------
        RN.Vector instance

        Note
        ----
        While the space has a single (unique) zero vector,
        each call to this method returns a new instance of this vector.

        Examples
        --------

        >>> rn = RN(3)
        >>> x = rn.zero()
        >>> x
        RN(3).element([ 0.,  0.,  0.])
        """
        return self.element(np.zeros(self._dim, dtype=np.float64))

    @property
    def field(self):
        """ The underlying field of RN is the real numbers

        Parameters
        ----------
        None

        Returns
        -------
        RealNumbers instance

        Examples
        --------

        >>> rn = RN(3)
        >>> rn.field
        RealNumbers()
        """
        return self._field

    @property
    def dim(self):
        """ The number of dimensions of this space

        Parameters
        ----------
        None

        Returns
        -------
        RealNumbers instance

        Examples
        --------

        >>> rn = RN(3)
        >>> rn.dim
        3
        """
        return self._dim

    def equals(self, other):
        """ Verifies that other is a RN instance of dimension `n`

        Parameters
        ----------
        other : any object
                The object to check for equality

        Returns
        -------
        boolean      True or false

        Examples
        --------

        Comparing with self
        >>> r3 = RN(3)
        >>> r3.equals(r3)
        True

        Also true when comparing with similar instance
        >>> r3a, r3b = RN(3), RN(3)
        >>> r3a.equals(r3b)
        True

        False when comparing to other dimension RN
        >>> r3, r4 = RN(3), RN(4)
        >>> r3.equals(r4)
        False

        We also support operators '==' and '!='
        >>> r3, r4 = RN(3), RN(4)
        >>> r3 == r3
        True
        >>> r3 == r4
        False
        >>> r3 != r4
        True
        """
        return isinstance(other, RN) and self.dim == other.dim

    def __str__(self):
        return "RN(" + str(self.dim) + ")"

    def __repr__(self):
        return 'RN(' + str(self.dim) + ')'

    class Vector(LinearSpace.Vector):
        """ A RN-vector represented using numpy

        Parameters
        ----------

        space : RN
                Instance of RN this vector lives in
        data : numpy.ndarray
               Underlying data-representation to be used by this vector
               The dtype of the array must be float64
               The shape of the array must be (n,)
        """

        def __init__(self, space, data):
            if not isinstance(data, np.ndarray):
                raise TypeError(errfmt('''
                'data' ({}) must be a numpy.ndarray
                '''.format(type(data))))

            if data.dtype != np.float64:
                raise TypeError(errfmt('''
                type('data') ({}) must be float64
                '''.format(data.dtype)))

            super().__init__(space)
            self._data = data

        @property
        def data(self):
            """
            Get the underlying datacontainer, an numpy-array

            Parameters
            ----------
            None

            Returns
            -------
            data : Numpy.ndarray
                   The underlying data representation

            Examples
            --------
            >>> vec = RN(3).element([1, 2, 3])
            >>> vec.data
            array([ 1.,  2.,  3.])
            """
            return self._data

        @property
        def data_ptr(self):
            """
            Get a pointer to the underlying data.

            Parameters
            ----------
            None

            Returns
            -------
            data : Numpy.ndarray
                   The underlying data representation

            Examples
            --------
            >>> import ctypes
            >>> vec = RN(3).element([1, 2, 3])
            >>> ArrayType = ctypes.c_double*3
            >>> arr = np.frombuffer(ArrayType.from_address(vec.data_ptr))
            >>> arr
            array([ 1.,  2.,  3.])

            Inplace modifications

            >>> arr[0] = 5
            >>> vec
            RN(3).element([ 5.,  2.,  3.])
            """
            return self._data.ctypes.data

        def __str__(self):
            return str(self.data)

        def __repr__(self):
            val_str = repr(self.data).lstrip('array(').rstrip(')')
            return repr(self.space) + '.element(' + val_str + ')'

        def __len__(self):
            """ Get the dimension of the underlying space

            Parameters
            ----------
            None

            Returns
            -------
            len : int

            Examples
            --------
            >>> RN(3).element().__len__()
            3
            >>> len(RN(3).element())
            3
            """
            return self.space.dim

        def __getitem__(self, index):
            """ Access values of this vector

            Parameters
            ----------

            index : int or slice
                    The position(s) that should be accessed

            Returns
            -------
            If index is an `int`
            numpy.float64, value at index

            If index is an `slice`
            numpy.ndarray instance with the values at the slice


            Examples
            --------

            >>> rn = RN(3)
            >>> y = rn.element([1, 2, 3])
            >>> y[0]
            1.0
            >>> y[1:3]
            array([ 2.,  3.])

            """
            return self.data.__getitem__(index)

        def __setitem__(self, index, value):
            """ Set values of this vector

            Parameters
            ----------

            index : int or slice
                    The position(s) that should be set
            value : float or Array-Like
                    The values that should be assigned.
                    If index is an integer, value should be a float.
                    If index is a slice, value should be an Array-Like
                    of the same size as the slice.

            Returns
            -------
            None


            Examples
            --------

            >>> rn = RN(3)
            >>> y = rn.element([1, 2, 3])
            >>> y[0] = 5
            >>> y
            RN(3).element([ 5.,  2.,  3.])
            >>> y[1:3] = [7, 8]
            >>> y
            RN(3).element([ 5.,  7.,  8.])
            >>> y[:] = np.array([0, 0, 0])
            >>> y
            RN(3).element([ 0.,  0.,  0.])

            """

            return self.data.__setitem__(index, value)


class NormedRN(RN, NormedSpace):
    """ The real space R^n with the p-norm.

    Parameters
    ----------

    n : int
        The dimension of the space
    ord : float
          The order of the norm

    Notes
    -----

    The following values for `ord` can be specified.
    Note that any value of ord < 1 only gives a pseudonorm.

    =====  ====================================================
    ord    Definition
    =====  ====================================================
    inf    max(norm(x[0]), ..., norm(x[n-1]))
    -inf   min(norm(x[0]), ..., norm(x[n-1]))
    0      (norm(x[0]) != 0 + ... + norm(x[n-1]) != 0)
    other  (norm(x[0])**ord + ... + norm(x[n-1])**ord)**(1/ord)
    =====  ====================================================
    """

    def __init__(self, n, ord=None):
        self.ord = ord if ord is not None else 2

        super().__init__(n)

    def _norm(self, vector):
        """ Calculates the p-norm of a vector

        Parameters
        ----------

        vector : NormedRN.Vector

        Returns
        -------
        norm : float
               Norm of the vector

        Examples
        --------

        >>> rn = NormedRN(2, ord=2)
        >>> x = rn.element([3, 4])
        >>> rn.norm(x)
        5.0


        >>> rn = NormedRN(2, ord=1)
        >>> x = rn.element([3, 4])
        >>> rn.norm(x)
        7.0

        >>> rn = NormedRN(2, ord=0)
        >>> x = rn.element([3, 0])
        >>> rn.norm(x)
        1.0

        """

        # Use numpy norm
        return np.linalg.norm(vector.data, ord=self.ord)

    class Vector(RN.Vector, NormedSpace.Vector):
        """ A NormedRN-vector represented using numpy

        Parameters
        ----------

        space : RN
                Instance of RN this vector lives in
        data : numpy.ndarray
               Underlying data-representation to be used by this vector
               The dtype of the array must be float64
               The shape of the array must be (n,)
        """


class EuclideanSpace(RN, HilbertSpace, Algebra):
    """The real space R^n with the usual inner product.

    Parameters
    ----------

    n : int
        The dimension of the space
    """

    def __init__(self, n):
        super().__init__(n)

        self._dot, self._nrm2 = get_blas_funcs(['dot', 'nrm2'])

    def _norm(self, x):
        """ Calculates the norm of a vector.

        This is defined as:

        norm(x) := sqrt(x[0]**2 + x[1]**2 + ... x[n-1]**2)

        Parameters
        ----------

        x : EuclideanSpace.Vector

        Returns
        -------
        norm : float
               Norm of the vector

        Examples
        --------

        >>> rn = EuclideanSpace(2)
        >>> x = rn.element([3, 4])
        >>> rn.norm(x)
        5.0

        """
        # TODO: Possibly change this to 'dot', which is faster
        return float(self._nrm2(x.data))

    def _inner(self, x, y):
        """ Calculates the inner product of two vectors

        This is defined as:

        inner(x,y) := x[0]*y[0] + x[1]*y[1] + ... x[n-1]*y[n-1]

        Parameters
        ----------

        x : EuclideanSpace.Vector
        y : EuclideanSpace.Vector

        Returns
        -------
        inner : float
                Inner product of x and y.

        Examples
        --------

        >>> rn = EuclideanSpace(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> 5*1 + 3*2 + 2*3
        17
        >>> rn.inner(x, y)
        17.0

        """

        return float(self._dot(x.data, y.data))

    def _multiply(self, x, y):
        """ Calculates the pointwise product of two vectors and assigns the
        result to `y`

        This is defined as:

        multiply(x,y) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

        Parameters
        ----------

        x : RNVector
            read from
        y : RNVector
            read from and written to

        Returns
        -------
        None

        Examples
        --------

        >>> rn = EuclideanSpace(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> rn.multiply(x, y)
        >>> y
        EuclideanSpace(3).element([ 5.,  6.,  6.])
        """
        y.data[:] = x.data * y.data

    def __repr__(self):
        return 'EuclideanSpace(' + str(self.dim) + ')'

    class Vector(RN.Vector, HilbertSpace.Vector, Algebra.Vector):
        """ A EuclideanSpace-vector represented using numpy

        Parameters
        ----------

        space : RN
                Instance of RN this vector lives in
        data : numpy.ndarray
               Underlying data-representation to be used by this vector
               The dtype of the array must be float64
               The shape of the array must be (n,)
        """


if __name__ == '__main__':
    import doctest
    doctest.testmod()
