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

""" Module for R^n, the space of n-tuples of real numbers

This is the default implementation of R^n and the
corresponding MetricRn, NormedRn and EuclidRn.

The data is represented by NumPy arrays.
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import str, super
from future import standard_library

# External module imports
import numpy as np
from scipy.lib.blas import get_blas_funcs
from numbers import Integral, Real
from math import sqrt

# RL imports
from RL.space.space import (LinearSpace, MetricSpace, NormedSpace,
                            HilbertSpace, Algebra)
from RL.space.set import RealNumbers
from RL.utility.utility import errfmt
try:
    from RL.space.cuda import CudaRn
    try:
        CudaRn(1).element()
    except MemoryError:
        raise MemoryError("RLcpp seems to be broken.")
    CUDA_AVAILABLE = True
except ImportError:
    CudaRn = None
    CUDA_AVAILABLE = False

standard_library.install_aliases()


class Rn(LinearSpace):
    """The real space R^n without any further mathematical structure

    Parameters
    ----------

    dim : int
        The dimension of the space
    """

    def __init__(self, dim):
        if not isinstance(dim, Integral) or dim < 1:
            raise TypeError(errfmt('''
            'dim' ({}) has to be a positive integer'''.format(np)))
        self._dim = dim
        self._field = RealNumbers()
        self._axpy, self._scal, self._copy = get_blas_funcs(
            ['axpy', 'scal', 'copy'])

    def element(self, data=None, **kwargs):
        """ Create an element in R^n

        Parameters
        ----------
        data : array-like, optional
            The array of values to fill the new array with. It must
            be castable to a numpy.ndarray with dtype=float64 and
            shape=(n,).

        Returns
        -------
        element : Rn.Vector instance

        Note
        ----
        If called without arguments, the values of the returned vector
        may be _anything_ including inf and NaN. Thus, operations such
        as element() * 0 need not result in the zero vector.

        Examples
        --------

        >>> r3 = Rn(3)
        >>> x = r3.element()
        >>> x in r3
        True

        >>> x = r3.element([1, 2, 3])
        >>> x
        Rn(3).element([ 1.,  2.,  3.])

        Existing NumPy arrays are wrapped instead of copied if their
        dtype is float64:

        >>> a = np.array([1., 2., 3.])
        >>> x = r3.element(a)
        >>> x
        Rn(3).element([ 1.,  2.,  3.])
        >>> x.data is a
        True

        >>> b = np.array([1, 2, 3])
        >>> x = r3.element(b)
        >>> x
        Rn(3).element([ 1.,  2.,  3.])
        >>> x.data is b
        False

        """

        if data is None:
            data = np.empty(self.dim, dtype=np.float64)
        else:
            data = np.array(data, dtype=np.float64, copy=False)

            if data.shape != (self.dim,):
                raise ValueError(errfmt('''
                Input numpy array ({}) is of shape {}, expected shape {}
                '''.format(data, data.shape, (self.dim,))))

            if data.dtype != np.float64:
                raise ValueError(errfmt('''
                Input numpy array ({}) is of type {}, expected float64
                '''.format(data, data.dtype)))

        return self.Vector(self, data)

    def _lincomb(self, z, a, x, b, y):
        """ Implement z = a*x + b*y using optimized BLAS rutines

        Parameters
        ----------
        z : Rn.Vector
            The Vector that the result is written to.
        a : RealNumber
            Scalar to multiply `x` with.
        x : Rn.Vector
            The first of the summands
        b : RealNumber
            Scalar to multiply `y` with.
        y : Rn.Vector
            The second of the summands

        Returns
        -------
        None

        Examples
        --------
        >>> r3 = Rn(3)
        >>> x = r3.element([1, 2, 3])
        >>> y = r3.element([4, 5, 6])
        >>> z = r3.element()
        >>> r3.lincomb(z, 2, x, 3, y)
        >>> z
        Rn(3).element([ 14.,  19.,  24.])

        """

        if x is y and b != 0:
            # x is aligned with y -> z = (a+b)*x
            self._lincomb(z, a+b, x, 0, x)
        elif z is x and z is y:
            # All the vectors are aligned -> z = (a+b)*z
            self._scal(a+b, z.data)
        elif z is x:
            # z is aligned with x -> z = a*z + b*y
            if a != 1:
                self._scal(a, z.data)
            if b != 0:
                self._axpy(y.data, z.data, self.dim, b)
        elif z is y:
            # z is aligned with y -> z = a*x + b*z
            if b != 1:
                self._scal(b, z.data)
            if a != 0:
                self._axpy(x.data, z.data, self.dim, a)
        else:
            # We have exhausted all alignment options, so x != y != z
            # We now optimize for various values of a and b
            if b == 0:
                if a == 0:  # Zero assignment -> z = 0
                    z.data[:] = 0
                else:  # Scaled copy -> z = a*x
                    self._copy(x.data, z.data)
                    if a != 1:
                        self._scal(a, z.data)
            else:
                if a == 0:  # Scaled copy -> z = b*y
                    self._copy(y.data, z.data)
                    if b != 1:
                        self._scal(b, z.data)

                elif a == 1:  # No scaling in x -> z = x + b*y
                    self._copy(x.data, z.data)
                    self._axpy(y.data, z.data, self.dim, b)
                else:  # Generic case -> z = a*x + b*y
                    self._copy(y.data, z.data)
                    if b != 1:
                        self._scal(b, z.data)
                    self._axpy(x.data, z.data, self.dim, a)

    def zero(self):
        """ Create a vector of zeros

        Parameters
        ----------
        None

        Returns
        -------
        zero : Rn.Vector instance

        Examples
        --------

        >>> r3 = Rn(3)
        >>> x = r3.zero()
        >>> x
        Rn(3).element([ 0.,  0.,  0.])
        """

        return self.element(np.zeros(self._dim, dtype=np.float64))

    @property
    def field(self):
        """ The field of R^n, i.e. the real numbers

        Parameters
        ----------
        None

        Returns
        -------
        field : RealNumbers instance

        Examples
        --------

        >>> r3 = Rn(3)
        >>> r3.field
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
        dim: int

        Examples
        --------

        >>> r3 = Rn(3)
        >>> r3.dim
        3
        """

        return self._dim

    def equals(self, other):
        """ Check if `other` is an Rn instance of the same dimension

        Parameters
        ----------
        other : any object
            The object to check for equality

        Returns
        -------
        equals : boolean

        Examples
        --------

        >>> r3 = Rn(3)
        >>> r3.equals(r3)
        True

        Equality is not identity:

        >>> r3a, r3b = Rn(3), Rn(3)
        >>> r3a.equals(r3b)
        True
        >>> r3a is r3b
        False

        >>> r3, r4 = Rn(3), Rn(4)
        >>> r3.equals(r4)
        False

        Equality can also be checked with "==":

        >>> r3, r4 = Rn(3), Rn(4)
        >>> r3 == r3
        True
        >>> r3 == r4
        False
        >>> r3 != r4
        True
        """

        return isinstance(other, Rn) and self.dim == other.dim

    def __repr__(self):
        return 'Rn({})'.format(self.dim)

    def __str__(self):
        return self.__repr__()

    class Vector(LinearSpace.Vector):
        """ An R^n vector represented with a NumPy array

        Parameters
        ----------

        space : Rn
            Space instance this vector lives in
        data : numpy.ndarray
            Array that will be used as data representation. Its dtype
            must be float64, and its shape must be (n,).
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
            """ The vector's data representation, a numpy array

            Parameters
            ----------
            None

            Returns
            -------
            data : numpy.ndarray
                The underlying data representation

            Examples
            --------
            >>> vec = Rn(3).element([1, 2, 3])
            >>> vec.data
            array([ 1.,  2.,  3.])
            """

            return self._data

        @property
        def data_ptr(self):
            """ A raw pointer to the underlying data.

            Parameters
            ----------
            None

            Returns
            -------
            data_ptr : int
                The memory address of the data representation

            Examples
            --------
            >>> import ctypes
            >>> vec = Rn(3).element([1, 2, 3])
            >>> arr_type = ctypes.c_double*3
            >>> arr = np.frombuffer(arr_type.from_address(vec.data_ptr))
            >>> arr
            array([ 1.,  2.,  3.])

            In-place modification:

            >>> arr[0] = 5
            >>> vec
            Rn(3).element([ 5.,  2.,  3.])
            """
            return self._data.ctypes.data

        def __str__(self):
            return str(self.data)

        def __repr__(self):
            val_str = repr(self.data).lstrip('array(').rstrip(')')
            return '{!r}.element({})'.format(self.space, val_str)

        def __len__(self):
            """ The dimension of the underlying space

            Parameters
            ----------
            None

            Returns
            -------
            len : int

            Examples
            --------
            >>> len(Rn(3).element())
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
            value: float64 or numpy.ndarray
                The value(s) at the index (indices)


            Examples
            --------

            >>> rn = Rn(3)
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
            value : float or array-like
                The values that should be assigned.
                If 'index' is an integer, 'value' must be a float.
                If 'index' is a slice, 'value' must be broadcastable
                to the size of the slice (same size, shape (1,)
                or float).

            Returns
            -------
            None


            Examples
            --------

            >>> rn = Rn(3)
            >>> y = rn.element([1, 2, 3])
            >>> y[0] = 5
            >>> y
            Rn(3).element([ 5.,  2.,  3.])
            >>> y[1:3] = [7, 8]
            >>> y
            Rn(3).element([ 5.,  7.,  8.])
            >>> y[:] = np.array([0, 0, 0])
            >>> y
            Rn(3).element([ 0.,  0.,  0.])

            Broadcasting:

            >>> y[1:3] = -2.
            >>> y
            Rn(3).element([ 0., -2., -2.])
            """

            return self.data.__setitem__(index, value)


class MetricRn(Rn, MetricSpace):
    """ The real space R^n as a metric space without norm

    Parameters
    ----------

    dim : int
        The dimension of the space
    dist : callable
        The distance function defining a metric on R^n. It must accept
        two array arguments and return a nonnegative float.
    """

    def __init__(self, dim, dist):
        if not callable(dist):
            raise TypeError("'dist' must be callable.")

        self._custom_dist = dist

        super().__init__(dim)

    def _dist(self, x, y):
        return self._custom_dist(x, y)

    def __repr__(self):
        return 'MetricRn({})'.format(self.dim)

    def __str__(self):
        return self.__repr__()

    class Vector(Rn.Vector, MetricSpace.Vector):
        """ A MetricRn vector represented by a NumPy array

        Parameters
        ----------

        space : Rn
            Space instance this vector lives in
        data : numpy.ndarray
            Array that will be used as data representation. Its dtype
            must be float64, and its shape must be (dim,).
        """


class NormedRn(Rn, NormedSpace):
    """ The real space R^n with the p-norm or a custom norm.

    Parameters
    ----------

    dim : int
        The dimension of the space
    p : float, optional
        The order of the norm. Default: 2.0
    kwargs: {'weights', 'norm'}
            'weights': array-like, optional
                Array of weights to be used in the norm calculation.
                It must be broadcastable to size (n,). All entries
                must be positive.
                Cannot be combined with 'norm'.
            'norm': callable, optional
                A custom norm to use instead of the p-norm. Cannot be
                combined with 'p'.

    Notes
    -----

    The following values for `p` can be specified.
    Note that any value of p < 1 only gives a pseudonorm.

    =====  ====================================================
    p      Definition
    =====  ====================================================
    inf    max(abs(x[0]), ..., abs(x[n-1]))
    -inf   min(abs(x[0]), ..., abs(x[n-1]))
    0      (x[0] != 0 + ... + x[n-1] != 0)
    other  (abs(x[0])**p + ... + abs(x[n-1])**p)**(1/p)
    =====  ====================================================
    """

    def __init__(self, dim, p=None, **kwargs):
        weights = kwargs.get('weights', None)
        norm = kwargs.get('norm', None)

        if p is not None and norm is not None:
            raise ValueError(errfmt('''
            'p' and 'norm' cannot be combined.'''))

        p = p if p is not None else 2.0
        if not isinstance(p, Real):
            raise TypeError(errfmt('''
            'p' ({}) must be a real number.'''.format(p)))

        if weights is not None:
            if norm is not None:
                raise ValueError(errfmt('''
                'weights' and 'norm' cannot be combined.'''))
            try:
                weights = np.atleast_1d(weights)
            except TypeError:
                raise TypeError(errfmt('''
                'weights' ({}) must be array-like.'''.format(weights)))

            if not np.all(weights > 0):
                raise ValueError(errfmt('''
                'weights' must be all positive'''))

        if norm is not None and not callable(norm):
            raise TypeError("'norm' must be callable.")

        self._p = p
        self._sqrt_weights = np.sqrt(weights) if weights is not None else None
        self._custom_norm = norm

        super().__init__(dim)

    def _norm(self, vector):
        """ Calculate the norm of a vector

        Parameters
        ----------

        vector : NormedRn.Vector

        Returns
        -------
        norm : float
            Norm of the vector

        Examples
        --------

        >>> r2_2 = NormedRn(2, p=2)
        >>> x = r2_2.element([3, 4])
        >>> r2_2.norm(x)
        5.0

        >>> r2_1 = NormedRn(2, p=1)
        >>> x = r2_1.element([3, 4])
        >>> r2_1.norm(x)
        7.0

        >>> r2_0 = NormedRn(2, p=0)
        >>> x = r2_0.element([3, 0])
        >>> r2_0.norm(x)
        1.0

        Custom norm:

        >>> from functools import partial
        >>> norm_0 = partial(np.linalg.norm, ord=0)
        >>> r2_c = NormedRn(2, norm=norm_0)
        >>> x = r2_c.element([3, 0])
        >>> r2_c.norm(x) == r2_0.norm(x)
        True

        # TODO: weights doctest
        """

        if self._custom_norm is not None:
            return self._custom_norm(vector.data)
        elif self._sqrt_weights is None:
            return np.linalg.norm(vector.data, ord=self._p)
        else:
            return np.linalg.norm(vector.data * self._sqrt_weights,
                                  ord=self._p)

    def __repr__(self):
        return 'NormedRn({})'.format(self.dim)

    def __str__(self):
        return self.__repr__()

    class Vector(Rn.Vector, NormedSpace.Vector):
        """ A NormedRn vector represented by a NumPy array

        Parameters
        ----------

        space : NormedRn
            Space instance this vector lives in
        data : numpy.ndarray
            Array that will be used as data representation. Its dtype
            must be float64, and its shape must be (n,).
        """


class EuclidRn(Rn, HilbertSpace, Algebra):
    """The real space R^n with the usual inner (dot) product.

    Parameters
    ----------

    dim : int
        The dimension of the space
    """

    def __init__(self, dim, **kwargs):
        weights = kwargs.get('weights', None)
        inner = kwargs.get('inner', None)

        if weights is not None:
            if inner is not None:
                raise ValueError(errfmt('''
                'weights' and 'inner' cannot be combined.'''))
            try:
                weights = np.atleast_1d(weights)
            except TypeError:
                raise TypeError(errfmt('''
                'weights' ({}) must be array-like.'''.format(weights)))

            if not np.all(weights > 0):
                raise ValueError(errfmt('''
                'weights' must be all positive'''))

        if inner is not None and not callable(inner):
            raise TypeError("'inner' must be callable.")

        self._weights = weights
        self._custom_inner = inner
        self._dot = get_blas_funcs(['dot'])[0]

        super().__init__(dim)

    def _norm(self, x):
        """ Calculate the norm of a vector.

        norm(x) := sqrt(inner(x, x)).

        Parameters
        ----------

        x : EuclidRn.Vector

        Returns
        -------
        norm : float
               Norm of the vector

        Examples
        --------

        >>> rn = EuclidRn(2)
        >>> x = rn.element([3, 4])
        >>> rn.norm(x)
        5.0

        """

        return sqrt(self._inner(x, x))

    def _inner(self, x, y):
        """ Calculate the inner product of two vectors

        This is defined as:

        inner(x,y) := x[0]*y[0] + x[1]*y[1] + ... x[n-1]*y[n-1]

        Parameters
        ----------

        x : EuclidRn.Vector
        y : EuclidRn.Vector

        Returns
        -------
        inner : float
            Inner product of x and y.

        Examples
        --------

        >>> rn = EuclidRn(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> rn.inner(x, y) == 5*1 + 3*2 + 2*3
        True

        TODO: weighted / custom
        """

        if self._custom_inner is not None:
            return self._custom_inner(x.data, y.data)
        elif self._weights is None:
            return float(self._dot(x.data, y.data))
        else:
            return float(self._dot(x.data, self._weights * y.data))

    def _multiply(self, x, y):
        """ Calculate the pointwise product of two vectors and assign the
        result to `y`

        This is defined as:

        multiply(x,y) := [x[0]*y[0], x[1]*y[1], ..., x[n-1]*y[n-1]]

        Parameters
        ----------

        x : EuclidRn.Vector
            read from
        y : EuclidRn.Vector
            read from and written to

        Returns
        -------
        None

        Examples
        --------

        >>> rn = EuclidRn(3)
        >>> x = rn.element([5, 3, 2])
        >>> y = rn.element([1, 2, 3])
        >>> rn.multiply(x, y)
        >>> y
        EuclidRn(3).element([ 5.,  6.,  6.])
        """

        y.data[:] = x.data * y.data

    def __repr__(self):
        return 'EuclidRn({})'.format(self.dim)

    def __str__(self):
        return self.__repr__()

    class Vector(Rn.Vector, HilbertSpace.Vector, Algebra.Vector):
        """ A EuclidRn-vector represented using numpy

        Parameters
        ----------

        space : EuclidRn
            Space instance this vector lives in
        data : numpy.ndarray
            Array that will be used as data representation. Its dtype
            must be float64, and its shape must be (dim,).
        """


def rn(dim, impl='numpy', **kwargs):
    """ Create an R^n, by default the Euclidean space

    TODO: doc
    """

    try:
        impl = impl.lower()
    except AttributeError:
        raise TypeError("'impl' must be a string")

    dist = kwargs.get('dist', None)
    norm = kwargs.get('norm', None)
    norm_p = kwargs.get('norm_p', None)
    inner = kwargs.get('inner', None)
    weights = kwargs.get('weights', None)

    # Check if the parameter combination makes sense. The checks for
    # correct types or values are delegated to the class initializers
    if impl == 'numpy':
        # 'dist' is processed first since int short-cuts to 'Rn' or
        # 'MetricRn' if provided
        if dist is False:
            return Rn(dim)
        elif dist is not None:
            if norm is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'norm' '''))
            if norm_p is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'norm_p' '''))
            if inner is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'inner' '''))
            if weights is not None:
                raise ValueError(errfmt('''
                'dist' cannot be combined with 'weights' '''))

            return MetricRn(dim, dist=dist)

        # 'dist' not specified, continuing with 'norm' and 'norm_p'
        if norm is not None and weights is not None:
            raise ValueError(errfmt('''
            'norm' cannot be combined with 'weights' '''))
        elif norm is not None or norm_p is not None:
            if inner is not None:
                raise ValueError(errfmt('''
                'norm' or 'norm_p' cannot be combined with 'inner' '''))

            return NormedRn(dim, norm=norm, norm_p=norm_p, weights=weights)
        else:
            # neither 'dist' nor 'norm' nor 'norm_p' specified,
            # assuming inner product space
            if inner is not None and weights is not None:
                raise ValueError(errfmt('''
                'inner' cannot be combined with 'weights' '''))
            return EuclidRn(dim, inner=inner, weights=weights)

    if impl == 'cuda':
        if not CUDA_AVAILABLE:
            raise ValueError(errfmt('''
            CUDA implementation not available'''))

        # TODO: move to CudaRn.__init__
        if norm_p is not None:
            raise NotImplementedError(errfmt('''
            p-norms for p != 2.0 in CUDA spaces not implemented'''))

        if norm is not None:
            raise ValueError(errfmt('''
            Custom norm implementation not possible for CUDA spaces'''))
        if inner is not None:
            raise ValueError(errfmt('''
            Custom inner product implementation not possible for CUDA
            spaces'''))

        # TODO: move to CudaRn.__init__
        if weights is not None:
            raise NotImplementedError(errfmt('''
            Weighted CUDA spaces not implemented'''))

        return CudaRn(dim)

    else:
        raise ValueError(errfmt('''
        Invalid value {} for 'impl' '''.format(impl)))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
