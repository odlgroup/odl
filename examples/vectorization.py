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

"""Example showing how to use vectorization of FunctionSpaceVector's."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
import numpy as np
import odl
import timeit


def performace_example():
    # Create a function space
    X = odl.FunctionSpace(odl.Interval(0, 1))

    # Functions default to vectorized
    f_vec = X.element(lambda x: x**2)

    # If 'vectorized=False' is used, odl automatically vectorizes
    f_novec = X.element(lambda x: x**2, vectorized=False)

    # Example of runtime, expect vectorized to be much faster
    points = np.linspace(0, 1, 10000)

    print('Vectorized runtime:     {:5f}'
          ''.format(timeit.timeit(lambda: f_vec(points), number=100)))
    print('Non-vectorized runtime: {:5f}'
          ''.format(timeit.timeit(lambda: f_novec(points), number=100)))


def numba_example():
    # Some functions are not easily vectorized,
    # here we can use numba to improve performance.

    try:
        import numba
    except ImportError:
        print('Numba not installed, skipping.')
        return

    def myfunc(x):
        "Return a-b if a>b, otherwise return a+b"
        if x[0] > x[1]:
            return x[0] - x[1]
        else:
            return x[0] + x[1]

    my_vectorized_func = numba.vectorize(myfunc)

    # Create functions
    X = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))
    f_default = X.element(myfunc, vectorized=False)
    f_numba = X.element(my_vectorized_func)

    # Create points
    points = odl.uniform_sampling(X.domain, [100, 100]).points().T

    print('Vectorized runtime:     {:5f}'
          ''.format(timeit.timeit(lambda: f_default(points), number=100)))
    print('Non-vectorized runtime: {:5f}'
          ''.format(timeit.timeit(lambda: f_numba(points), number=100)))

if __name__ == '__main__':
    print('Running performance example')
    performace_example()

    print('Running numba example')
    numba_example()
