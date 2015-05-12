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
try:
    from builtins import str, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, super
from future import standard_library

# External module imports
import numpy
from scipy.lib.blas import get_blas_funcs

# RL imports
from RL.space.space import *
from RL.space.set import *
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
        self._n = n
        self._field = RealNumbers()
        self._axpy, self._scal, self._copy = get_blas_funcs(['axpy',
                                                             'scal',
                                                             'copy'])

    def linCombImpl(self, z, a, x, b, y):
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
        >>> x = rn.makeVector([1, 2, 3])
        >>> y = rn.makeVector([4, 5, 6])
        >>> z = rn.empty()
        >>> rn.linComb(z, 2, x, 3, y)
        >>> z
        RN(3).makeVector([ 14.,  19.,  24.])

        """

        if x is y and b != 0:
            # If x is aligned with y, we are looking at:     z = (a+b)*x
            self.linCombImpl(z, a+b, x, 0, x)
        elif z is x and z is y:
            # If all the vectors are aligned we have:        z = (a+b)*z
            self._scal(a+b, z.values)
        elif z is x:
            # If z is aligned with x we have                 z = a*z + b*y
            if a != 1:
                self._scal(a, z.values)
            if b != 0:
                self._axpy(y.values, z.values, self._n, b)
        elif z is y:
            # If z is aligned with y we have                 z = a*x + b*z
            if b != 1:
                self._scal(b, z.values)
            if a != 0:
                self._axpy(x.values, z.values, self._n, a)
        else:
            # We have exhausted all alignment options, so x != y != z
            # We now optimize for various values of a and b
            if b == 0:
                if a == 0:  # Zero assignment                z = 0
                    z.values[:] = 0
                else:                                       # z = a*x
                    self._copy(x.values, z.values)
                    if a != 1:
                        self._scal(a, z.values)
            else:
                if a == 0:                                  # z = b*y
                    self._copy(y.values, z.values)
                    if b != 1:
                        self._scal(b, z.values)

                elif a == 1:                                # z = x + b*y
                    self._copy(x.values, z.values)
                    self._axpy(y.values, z.values, self._n, b)
                else:                                       # z = a*x + b*y
                    self._copy(y.values, z.values)
                    if b != 1:
                        self._scal(b, z.values)
                    self._axpy(x.values, z.values, self._n, a)

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
        RN(3).makeVector([ 0.,  0.,  0.])
        """
        return self.makeVector(numpy.zeros(self._n, dtype=float))

    def empty(self):
        """ Returns an arbitrary vector

        more efficient than zeros.

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
        >>> x = rn.empty()
        >>> x in rn
        True

        """
        return self.makeVector(numpy.empty(self._n, dtype=float))

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
    def n(self):
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
        >>> rn.n
        3
        """
        return self._n

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
        return isinstance(other, RN) and self._n == other._n

    def makeVector(self, *args, **kwargs):
        """ Creates an element in RN

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
        >>> x = rn.makeVector(numpy.array([1., 2., 3.]))
        >>> x
        RN(3).makeVector([ 1.,  2.,  3.])
        >>> y = rn.makeVector([1, 2, 3])
        >>> y
        RN(3).makeVector([ 1.,  2.,  3.])

        """
        if isinstance(args[0], numpy.ndarray):
            if args[0].shape != (self._n,):
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape shape {}
                '''.format(args[0], args[0].shape, (self.n,))))

            if args[0].dtype != numpy.float64:
                raise ValueError(errfmt('''
                Input numpy array ({}) is of type {}, expected float64
                '''.format(args[0], args[0].dtype)))

            return self.Vector(self, args[0])
        else:
            return self.makeVector(
                numpy.array(*args, **kwargs).astype(numpy.float64, copy=False))

    def __str__(self):
        return "RN(" + str(self.n) + ")"

    def __repr__(self):
        return 'RN(' + str(self.n) + ')'

    class Vector(LinearSpace.Vector):
        """ A RN-vector represented using numpy

        Parameters
        ----------

        space : RN
                Instance of RN this vector lives in
        values : numpy.ndarray
                 Underlying data-representation to be used by this vector
                 The dtype of the array must be float64
                 The shape of the array must be (n,)
        """

        def __init__(self, space, values):
            super().__init__(space)
            self.values = values

        def __str__(self):
            return str(self.values)

        def __repr__(self):
            val_str = repr(self.values).lstrip('array(').rstrip(')')
            return repr(self.space) + '.makeVector(' + val_str + ')'

        def __len__(self):
            """ Get the dimension of the underlying space
            """
            return self.space.n

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
            >>> y = rn.makeVector([1, 2, 3])
            >>> y[0]
            1.0
            >>> y[1:3]
            array([ 2.,  3.])

            """
            return self.values.__getitem__(index)

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
            >>> y = rn.makeVector([1, 2, 3])
            >>> y[0] = 5
            >>> y
            RN(3).makeVector([ 5.,  2.,  3.])
            >>> y[1:3] = [7, 8]
            >>> y
            RN(3).makeVector([ 5.,  7.,  8.])
            >>> y[:] = numpy.array([0, 0, 0])
            >>> y
            RN(3).makeVector([ 0.,  0.,  0.])

            """

            return self.values.__setitem__(index, value)


class EuclidianSpace(RN, HilbertSpace, Algebra):
    """The real space R^n with the usual inner product.

    Parameters
    ----------

    n : int
        The dimension of the space
    """

    def __init__(self, n):
        super().__init__(n)

        self._dot, self._nrm2 = get_blas_funcs(['dot',
                                                'nrm2'])

    def innerImpl(self, x, y):
        """ Calculates the inner product of two vectors

        This is defined as:

        inner(x,y) := x[0]*y[0] + x[1]*y[1] + ... x[n-1]*y[n-1]

        Parameters
        ----------

        x : Vector
        y : Vector

        Returns
        -------
        float

        Examples
        --------

        >>> rn = EuclidianSpace(3)
        >>> x = rn.makeVector([5, 3, 2])
        >>> y = rn.makeVector([1, 2, 3])
        >>> 5*1 + 3*2 + 2*3
        17
        >>> rn.inner(x, y)
        17.0

        """
        return float(self._dot(x.values, y.values))

    def multiplyImpl(self, x, y):
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

        >>> rn = EuclidianSpace(3)
        >>> x = rn.makeVector([5, 3, 2])
        >>> y = rn.makeVector([1, 2, 3])
        >>> rn.multiply(x, y)
        >>> y
        EuclidianSpace(3).makeVector([ 5.,  6.,  6.])
        """
        y.values[:] = x.values*y.values

    def __repr__(self):
        return 'EuclidianSpace(' + str(self.n) + ')'

    def __repr__(self):
        return 'EuclidianSpace(' + str(self.n) + ')'
    
    class Vector(RN.Vector, HilbertSpace.Vector, Algebra.Vector):
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()
