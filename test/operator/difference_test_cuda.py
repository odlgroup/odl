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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()


# External module imports
import unittest

# ODL imports
import odl
from odl.util.testutils import ODLTestCase

try:
    import odlpp.odlpp_cuda as cuda
except ImportError:
    cuda = None


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available.')
class ForwardDiff(odl.LinearOperator):
    def __init__(self, space):
        if not isinstance(space, odl.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = self.range = space

    def _apply(self, rhs, out):
        cuda.forward_diff(rhs.data, out.data)

    @property
    def adjoint(self):
        return ForwardDiffAdjoint(self.domain)


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available.')
class ForwardDiffAdjoint(odl.LinearOperator):
    def __init__(self, space):
        if not isinstance(space, odl.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = self.range = space

    def _apply(self, rhs, out):
        cuda.forward_diff_adj(rhs.data, out.data)

    @property
    def adjoint(self):
        return ForwardDiff(self.domain)


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available.')
class ForwardDiff2D(odl.LinearOperator):
    def __init__(self, space):
        if not isinstance(space, odl.CudaRn):
            raise TypeError("space must be CudaPixelDiscretization")

        self.domain = space
        self.range = odl.ProductSpace(space, space)

    def _apply(self, rhs, out):
        cuda.forward_diff_2d(rhs.data, out[0].data, out[1].data,
                             self.domain.shape[0], self.domain.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2DAdjoint(self.domain)


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available.')
class ForwardDiff2DAdjoint(odl.LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, odl.CudaRn):
            raise TypeError("space must be CudaPixelDiscretization")

        self.domain = odl.ProductSpace(space, space)
        self.range = space

    def _apply(self, rhs, out):
        cuda.forward_diff_2d_adj(rhs[0].data, rhs[1].data, out.data,
                                 self.range.shape[0], self.range.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2D(self.range)


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available.')
class TestCudaForwardDifference(ODLTestCase):
    @unittest.skip('TODO: update to new discretization')
    def test_fwd_diff(self):
        # Continuous definition of problem
        space = odl.L2(odl.Interval(0, 1))

        # Discretization
        n = 6
        d = odl.l2_uniform_discretization(space, n, impl='cuda')
        fun = d.element([1, 2, 5, 3, 2, 1])

        # Create operator
        diff = ForwardDiff(d)

        self.assertAllAlmostEquals(diff(fun), [0, 3, -2, -1, -1, 0])
        self.assertAllAlmostEquals(diff.T(fun), [0, -1, -3, 2, 1, 0])
        self.assertAllAlmostEquals(diff.T(diff(fun)), [0, -3, 5, -1, 0, 0])


@unittest.skipIf(not odl.CUDA_AVAILABLE, 'CUDA not available.')
class TestCudaForwardDifference2D(ODLTestCase):
    @unittest.skip('TODO: update to new discretization')
    def test_square(self):
        # Continuous definition of problem
        space = odl.L2(odl.Rectangle([0, 0], [1, 1]))

        # Discretization
        n = 5
        m = 5
        d = odl.l2_uniform_discretization(space, (n, m), impl='cuda')
        x, y = d.points()
        fun = d.element([[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

        diff = ForwardDiff2D(d)
        derivative = diff(fun)
        self.assertAllAlmostEquals(derivative[0][:].reshape(n, m),
                                   [[0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, -1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]])

        self.assertAllAlmostEquals(derivative[1][:].reshape(n, m),
                                   [[0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, -1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0]])

        # Verify that the adjoint is ok
        # -gradient.T(gradient(x)) is the laplacian
        laplacian = -diff.T(derivative)
        self.assertAllAlmostEquals(laplacian[:].reshape(n, m),
                                   [[0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 1, -4, 1, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0]])

    @unittest.skip('TODO: update to new discretization')
    def test_rectangle(self):
        # Continuous definition of problem
        space = odl.L2(odl.Rectangle([0, 0], [1, 1]))

        # Complicated functions to check performance
        n = 5
        m = 7

        # Discretization
        d = odl.l2_uniform_discretization(space, (n, m), impl='cuda')
        x, y = d.points()
        fun = d.element([[0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0]])

        diff = ForwardDiff2D(d)
        derivative = diff(fun)

        self.assertAllAlmostEquals(derivative[0][:].reshape(n, m),
                                   [[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, -1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]])

        self.assertAllAlmostEquals(derivative[1][:].reshape(n, m),
                                   [[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, -1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]])

        # Verify that the adjoint is ok
        # -gradient.T(gradient(x)) is the laplacian
        laplacian = -diff.T(derivative)
        self.assertAllAlmostEquals(laplacian[:].reshape(n, m),
                                   [[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [0, 1, -4, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]])


if __name__ == '__main__':
    unittest.main(exit=False)
