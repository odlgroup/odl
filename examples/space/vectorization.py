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

"""Example showing how to use vectorization of FunctionSpaceVector's."""

import numpy as np
import odl
import timeit


def performace_example():
    # Create a space of functions on the interval [0, 1].
    fspace = odl.FunctionSpace(odl.Interval(0, 1))

    # Simple function, already supports vectorization.
    f_vec = fspace.element(lambda x: x ** 2)

    # If 'vectorized=False' is used, odl automatically vectorizes with
    # the help of numpy.vectorize. This will be very slow, though, since
    # the implementation is basically a Python loop.
    f_novec = fspace.element(lambda x: x ** 2, vectorized=False)

    # We test both versions with 10000 evaluation points. The natively
    # vectorized version should be much faster than the one using
    # numpy.vectorize.
    points = np.linspace(0, 1, 10000)

    print('Vectorized runtime:     {:5f}'
          ''.format(timeit.timeit(lambda: f_vec(points), number=100)))
    print('Non-vectorized runtime: {:5f}'
          ''.format(timeit.timeit(lambda: f_novec(points), number=100)))


def numba_example():
    # Some functions are not easily vectorized, here we can use Numba to
    # improve performance.
    # See http://numba.pydata.org/

    try:
        import numba
    except ImportError:
        print('Numba not installed, skipping.')
        return

    def myfunc(x):
        """Return x - y if x > y, otherwise return x + y."""
        if x[0] > x[1]:
            return x[0] - x[1]
        else:
            return x[0] + x[1]

    # Numba expects functions f(x1, x2, x3, ...), while we have the
    # convention f(x) with x = (x1, x2, x3, ...). Therefore we need
    # to wrap the Numba-vectorized function.
    vectorized = numba.vectorize(lambda x, y: x - y if x > y else x + y)

    def myfunc_vec(x):
        """Return x - y if x > y, otherwise return x + y."""
        return vectorized(x[0], x[1])

    def myfunc_native_vec(x):
        """Return x - y if x > y, otherwise return x + y."""
        # This implementation uses Numpy's fast built-in vectorization
        # directly. The function np.where checks the condition in the
        # first argument and takes the values from the second argument
        # for all entries where the condition is `True`, otherwise
        # the values from the third argument are taken. The arrays are
        # automatically broadcast, i.e. the broadcast shape of the
        # condition expression determines the output shape.
        return np.where(x[0] > x[1], x[0] - x[1], x[0] + x[1])

    # Create (continuous) functions in the space of function defined
    # on the rectangle [0, 1] x [0, 1].
    fspace = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    f_default = fspace.element(myfunc, vectorized=False)
    f_numba = fspace.element(myfunc_vec)
    f_native = fspace.element(myfunc_native_vec, vectorized=True)

    # Create a unform grid in [0, 1] x [0, 1] (fspace.domain) with 2000
    # samples per dimension.
    grid = odl.uniform_sampling_fromintv(fspace.domain, [2000, 2000])
    # The points() method really creates all grid points (2000^2) and
    # stores them one-by-one (row-wise) in a large array with shape
    # (2000*2000, 2). Since the function expects points[i] to be the
    # array of i-th components of all points, we need to transpose.
    points = grid.points().T
    # The meshgrid property only returns a sparse representation of the
    # grid, a tuple whose i-th entry is the vector of all possible i-th
    # components in the grid (2000). Extra dimensions are added to the
    # vector in order to support automatic broadcasting. This is both
    # faster and more memory-friendly than creating the full point array.
    # See the numpy.meshgrid function for more information.
    mesh = grid.meshgrid  # Returns a sparse meshgrid (2000 * 2)

    print('Non-Vectorized runtime (points):      {:5f}'
          ''.format(timeit.timeit(lambda: f_default(points), number=1)))
    print('Non-Vectorized runtime (meshgrid):    {:5f}'
          ''.format(timeit.timeit(lambda: f_default(mesh), number=1)))
    print('Numba vectorized runtime (points):    {:5f}'
          ''.format(timeit.timeit(lambda: f_numba(points), number=1)))
    print('Numba vectorized runtime (meshgrid):  {:5f}'
          ''.format(timeit.timeit(lambda: f_numba(mesh), number=1)))
    print('Native vectorized runtime (points):   {:5f}'
          ''.format(timeit.timeit(lambda: f_native(points), number=1)))
    print('Native vectorized runtime (meshgrid): {:5f}'
          ''.format(timeit.timeit(lambda: f_native(mesh), number=1)))


if __name__ == '__main__':
    print('Running vectorization performance example.')
    performace_example()

    print('Running Numba example.')
    numba_example()
