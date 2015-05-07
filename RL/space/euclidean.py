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
import numpy as np
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
        """
        return self.makeVector(np.zeros(self._n, dtype=float))

    def empty(self):
        """ Returns an arbitrary vector

        more efficient than zeros.
        """
        return self.makeVector(np.empty(self._n, dtype=float))

    @property
    def field(self):
        """ The underlying field of RN is the real numbers
        """
        return self._field

    @property
    def n(self):
        """ The number of dimensions of this space
        """
        return self._n

    def equals(self, other):
        """ Verifies that other is a RN instance of dimension `n`
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
        >>> x = rn.makeVector(numpy.array([1,2,3]))
        >>> x
        [1.0, 2.0, 3.0]
        >>> y = rn.makeVector([1,2,3])
        >>> y
        [1.0, 2.0, 3.0]

        """
        if isinstance(args[0], np.ndarray):
            if args[0].shape != (self._n,):
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape shape {}
                '''.format(args[0], args[0].shape, (self.n,))))

            if args[0].dtype != np.float64:
                raise ValueError(errfmt('''
                Input numpy array ({}) is of type {}, expected float64
                '''.format(args[0], args[0].dtype)))

            return RN.Vector(self, args[0])
        else:
            return self.makeVector(
                np.array(*args, **kwargs).astype(np.float64, copy=False))

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self.n) + ")"

    def __repr__(self):
        return 'RN(' + str(self.n) + ')'

    class Vector(HilbertSpace.Vector, Algebra.Vector):
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
            return repr(self.space) + '.Vector(' + repr(self.values) + ')'

        def __len__(self):
            """ Get the dimension of the underlying space
            """
            return self.space._n

        def __getitem__(self, index):
            """ Access values of this vector

            Parameters
            ----------

            index : int or slice
                    The position(s) that should be accessed

            Returns
            -------
            If index is an `int`
            np.float64, value at index

            If index is an `slice`
            numpy.ndarray instance with the values at the slice


            Examples
            --------

            >>> rn = RN(3)
            >>> y = rn.makeVector([1, 2, 3])
            >>> y[0]
            1.0
            >>> y[1:2]
            array([2,3])

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
                    If index is a slice, value should be an Array-Like of the same
                    size as the slice.

            Returns
            -------
            None


            Examples
            --------

            >>> rn = RN(3)
            >>> y = rn.makeVector([1, 2, 3])
            >>> y[0] = 5
            >>> y
            [5.0, 2.0, 3.0]
            >>> y[1:2] = [7, 8]
            >>> y
            [5.0, 7.0, 8.0]
            >>> y[:] = numpy.array([0, 0, 0])
            >>> y
            [0.0, 0.0, 0.0]

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
        >>> rn.inner(x, y)
        17.0

        """
        return float(self._dot(x.values, y.values))

    def multiplyImpl(self, x, y):
        """ Calculates the pointwise product of two vectors and assigns the result to `y`

        This is defined as:

        multiply(x,y) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

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
        >>> rn.multiply(x, y)
        [5.0, 6.0, 6.0]
        """
        y.values[:] = x.values*y.values
