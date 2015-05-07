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
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
try:
    from builtins import str, super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import str, super
from future import standard_library
standard_library.install_aliases()


from math import sqrt

# External module imports
import numpy as np

# RL imports
import RL.operator.function as fun
import RL.space.space as spaces
import RL.space.set as sets
import RLcpp.PyCuda
# from RL.utility.utility import errfmt


class CudaRN(spaces.HilbertSpace, spaces.Algebra):
    """The real space R^n, implemented in CUDA

    Requires the compiled RL extension RLcpp.

    Parameters
    ----------

    n : int
        The dimension of the space
    """

    def __init__(self, n):
        self._n = n
        self._field = sets.RealNumbers()
        self.impl = RLcpp.PyCuda.CudaRNImpl(n)

    def innerImpl(self, x, y):
        return self.impl.inner(x.impl, y.impl)

    def normImpl(self, x):  # Optimized separately from inner
        return sqrt(self.impl.normSq(x.impl))

    def linCombImpl(self, z, a, x, b, y):
        self.impl.linComb(z.impl, a, x.impl, b, y.impl)

    def multiplyImpl(self, x, y):
        self.impl.multiply(x.impl, y.impl)

    def zero(self):
        """ Returns a vector of zeros
        """
        return self.makeVector(self.impl.zero())

    def empty(self):
        """ Returns a vector of zeros (CUDA memory is always initialized)
        """
        return self.makeVector(self.impl.empty())

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
        return isinstance(other, CudaRN) and self._n == other._n

    def makeVector(self, *args, **kwargs):
        """ Creates an element in CudaRN

        Parameters
        ----------
        The method has two call patter, the first is:

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
        CudaRN.Vector instance


        Examples
        --------

        >>> rn = CudaRN(3)
        >>> x = rn.makeVector(numpy.array([1,2,3]))
        >>> x
        [1.0, 2.0, 3.0]
        >>> y = rn.makeVector([1,2,3])
        >>> y
        [1.0, 2.0, 3.0]

        """

        if isinstance(args[0], RLcpp.PyCuda.CudaRNVectorImpl):
            return CudaRN.Vector(self, args[0])
        elif isinstance(args[0], np.ndarray): # Create from np array
            # Create result and assign (this could be optimized to one call)
            result = self.empty()
            result[:] = args[0]
            return result
        else:
            return self.makeVector(np.array(*args, **kwargs))

    def __str__(self):
        return "CudaRN(" + str(self._n) + ")"

    def __repr__(self):
        return "CudaRN(" + str(self._n) + ")"

    # These should likely be moved somewhere else!
    @property
    def abs(self):
        return fun.LambdaFunction(
            lambda input, output: RLcpp.PyCuda.abs(input.impl, output.impl),
            (self, self))

    @property
    def sign(self):
        return fun.LambdaFunction(
            lambda input, output: RLcpp.PyCuda.sign(input.impl, output.impl),
            input=(self, self))

    @property
    def addScalar(self):
        return fun.LambdaFunction(
            lambda input, scalar,
            output: RLcpp.PyCuda.addScalar(input.impl, scalar, output.impl),
            input=(self, self.field, self))

    @property
    def maxVectorScalar(self):
        return fun.LambdaFunction(
            lambda input, scalar,
            output: RLcpp.PyCuda.maxVectorScalar(input.impl, scalar,
                                                 output.impl),
            input=(self, self.field, self))

    @property
    def maxVectorVector(self):
        return fun.LambdaFunction(
            lambda input1, input2,
            output: RLcpp.PyCuda.maxVectorVector(input1.impl, input2.impl,
                                                 output.impl),
            input=(self, self, self))

    @property
    def sum(self):
        return fun.LambdaFunction(
            lambda input, output: RLcpp.PyCuda.abs(input.impl),
            input=(self), returns=self.field)

    class Vector(spaces.HilbertSpace.Vector, spaces.Algebra.Vector):
        """ A RN-vector represented in CUDA

        Parameters
        ----------

        space : CudaRN
                Instance of CudaRN this vector lives in
        values : RLcpp.PyCuda.CudaRNVectorImpl
                 Underlying data-representation to be used by this vector
        """
        def __init__(self, space, impl):
            super().__init__(space)
            self.impl = impl

        def __str__(self):
            return str(self[:])

        def __repr__(self):
            return repr(self.space) + '.Vector(' + str(self[:]) + ')'

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

            >>> rn = RN(3)
            >>> y = rn.makeVector([1, 2, 3])
            >>> y[0]
            1.0
            >>> y[1:2]
            array([2,3])

            """
            if isinstance(index, slice):
                return self.impl.getSlice(index)
            else:
                return self.impl.__getitem__(index)

        def __setitem__(self, index, value):
            """ Set values of this vector

            Parameters
            ----------

            index : int or slice
                    The position(s) that should be set
            value : float or Array-Like
                    The values that should be assigned.
                    If index is an integer, value should be a float.
                    If index is a slice, value should be an Array-Like of the
                    same size as the slice.

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

            if isinstance(index, slice):
                # Convert value to the correct type
                if not isinstance(value, np.ndarray):
                    value = np.array(value, dtype=np.float64)
                else:
                    # Cast to float if required (copy=False makes this a no-op
                    # if no cast is needed)
                    value = value.astype(np.float64, copy=False)

                # The impl checks that sizes match.
                self.impl.setSlice(index, value)
            else:
                self.impl.__setitem__(index, value)

    # End CudaRN.Vector
# End CudaRN
