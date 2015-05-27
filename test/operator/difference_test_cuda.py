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
from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import
from future import standard_library

# External module imports
import unittest

# RL imports
from RL.operator.operator import *
from RL.space.space import *
import RL.space.discretizations as dd
from RL.space.function import *
import RL.space.set as sets
from RL.space.product import productspace
from RL.utility.testutils import RLTestCase 

from RL.utility.testutils import RLTestCase, skip_all_tests, Timer

try:
    import RL.space.cuda as CS
    import RLcpp
except ImportError:
    RLTestCase = skip_all_tests("Missing RLcpp")

standard_library.install_aliases()


class ForwardDiff(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaRN")

        self.domain = self.range = space

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiff(rhs.data, out.data)

    @property
    def adjoint(self):
        return ForwardDiffAdjoint(self.domain)


class ForwardDiffAdjoint(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaRN")

        self.domain = self.range = space

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiffAdj(rhs.data, out.data)

    @property
    def adjoint(self):
        return ForwardDiff(self.domain)


class ForwardDiff2D(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaPixelDiscretization")

        self.domain = space
        self.range = productspace(space, space)

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiff2D(rhs.data, out[0].data, out[1].data,
                                 self.domain.cols, self.domain.rows)

    @property
    def adjoint(self):
        return ForwardDiff2DAdjoint(self.domain)


class ForwardDiff2DAdjoint(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaPixelDiscretization")

        self.domain = productspace(space, space)
        self.range = space

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiff2DAdj(rhs[0].data, rhs[1].data, out.data,
                                    self.range.cols, self.range.rows)

    @property
    def adjoint(self):
        return ForwardDiff2D(self.range)


class TestCudaForwardDifference(RLTestCase):
    def test_fwd_diff(self):
        # Continuous definition of problem
        I = sets.Interval(0, 1)
        space = L2(I)

        # Discretization
        n = 6
        rn = CS.CudaRN(n)
        d = dd.uniform_discretization(space, rn)
        fun = d.element([1, 2, 5, 3, 2, 1])

        # Create operator
        diff = ForwardDiff(d)

        self.assertAllAlmostEquals(diff(fun), [0, 3, -2, -1, -1, 0])
        self.assertAllAlmostEquals(diff.T(fun), [0, -1, -3, 2, 1, 0])
        self.assertAllAlmostEquals(diff.T(diff(fun)), [0, -3, 5, -1, 0, 0])


class TestCudaForwardDifference2D(RLTestCase):
    def test_square(self):
        # Continuous definition of problem
        I = sets.Rectangle([0, 0], [1, 1])
        space = L2(I)

        # Discretization
        n = 5
        m = 5
        rn = CS.CudaRN(n*m)
        d = dd.pixel_discretization(space, rn, n, m)
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

    def test_rectangle(self):
        # Continuous definition of problem
        I = sets.Rectangle([0, 0], [1, 1])
        space = L2(I)

        # Complicated functions to check performance
        n = 5
        m = 7

        # Discretization
        rn = CS.CudaRN(n*m)
        d = dd.pixel_discretization(space, rn, n, m)
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
