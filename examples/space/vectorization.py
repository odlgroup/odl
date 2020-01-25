"""Example showing how to use vectorization of `FunctionSpaceElement`'s."""

import numpy as np
import timeit

import odl
from odl.discr.discr_utils import make_func_for_sampling


def performance_example():
    # Simple function, already supports vectorization
    f_vec = make_func_for_sampling(
        lambda x: x ** 2, domain=odl.IntervalProd(0, 1)
    )

    # Vectorized with NumPy's poor man's vectorization function
    f_novec = np.vectorize(lambda x: x ** 2)

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

    def myfunc_numba(x):
        """Return x - y if x > y, otherwise return x + y."""
        return vectorized(x[0], x[1])

    def myfunc_vec(x):
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
    f_vec = make_func_for_sampling(
        myfunc_vec, domain=odl.IntervalProd([0, 0], [1, 1])
    )
    f_numba = make_func_for_sampling(
        myfunc_numba, domain=odl.IntervalProd([0, 0], [1, 1])
    )

    # Create a unform grid in [0, 1] x [0, 1] (fspace.domain) with 2000
    # samples per dimension.
    grid = odl.uniform_grid([0, 0], [1, 1], shape=(2000, 2000))
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

    print('Native vectorized runtime (points):   {:5f}'
          ''.format(timeit.timeit(lambda: f_vec(points), number=1)))
    print('Native vectorized runtime (meshgrid): {:5f}'
          ''.format(timeit.timeit(lambda: f_vec(mesh), number=1)))
    print('Numba vectorized runtime (points):    {:5f}'
          ''.format(timeit.timeit(lambda: f_numba(points), number=1)))
    print('Numba vectorized runtime (meshgrid):  {:5f}'
          ''.format(timeit.timeit(lambda: f_numba(mesh), number=1)))


if __name__ == '__main__':
    print('Running vectorization performance example.')
    performance_example()

    print('Running Numba example.')
    numba_example()
