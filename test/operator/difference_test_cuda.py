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
from builtins import super
from future import standard_library
standard_library.install_aliases()


# External module imports
import pytest

# ODL imports
import odl
from odl.util.testutils import all_almost_equal, skip_if_no_cuda

try:
    import odlpp.odlpp_cuda as cuda
except ImportError:
    cuda = None


class ForwardDiff(odl.Operator):
    def __init__(self, space):
        super().__init__(space, space, linear=True)

    def _apply(self, rhs, out):
        cuda.forward_diff(rhs.ntuple.data, out.ntuple.data)

    @property
    def adjoint(self):
        return ForwardDiffAdjoint(self.domain)


class ForwardDiffAdjoint(odl.Operator):
    def __init__(self, space):
        super().__init__(space, space, linear=True)

    def _apply(self, rhs, out):
        cuda.forward_diff_adj(rhs.ntuple.data, out.ntuple.data)

    @property
    def adjoint(self):
        return ForwardDiff(self.domain)


class ForwardDiff2D(odl.Operator):
    def __init__(self, space):
        super().__init__(space, odl.ProductSpace(space, space), linear=True)

    def _apply(self, rhs, out):
        cuda.forward_diff_2d(rhs.ntuple.data,
                             out[0].ntuple.data, out[1].ntuple.data,
                             rhs.shape[0], rhs.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2DAdjoint(self.domain)


class ForwardDiff2DAdjoint(odl.Operator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        super().__init__(odl.ProductSpace(space, space), space, linear=True)

    def _apply(self, rhs, out):
        cuda.forward_diff_2d_adj(rhs[0].ntuple.data, rhs[1].ntuple.data,
                                 out.ntuple.data,
                                 out.shape[0], out.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2D(self.range)


@skip_if_no_cuda
def test_fwd_diff():
    # Continuous definition of problem
    space = odl.FunctionSpace(odl.Interval(0, 1))

    # Discretization
    n = 6
    d = odl.uniform_discr(space, n, impl='cuda')
    fun = d.element([1, 2, 5, 3, 2, 1])

    # Create operator
    diff = ForwardDiff(d)

    assert all_almost_equal(diff(fun), [0, 3, -2, -1, -1, 0])
    assert all_almost_equal(diff.adjoint(fun), [0, -1, -3, 2, 1, 0])
    assert all_almost_equal(diff.adjoint(diff(fun)), [0, -3, 5, -1, 0, 0])


@skip_if_no_cuda
def test_square():
    # Continuous definition of problem
    space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))

    # Discretization
    n = 5
    m = 5
    d = odl.uniform_discr(space, (n, m), impl='cuda')

    fun = d.element([[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]])

    diff = ForwardDiff2D(d)
    derivative = diff(fun)
    assert all_almost_equal(derivative[0].asarray(),
                            [[0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, -1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])

    assert all_almost_equal(derivative[1].asarray(),
                            [[0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, -1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]])

    # Verify that the adjoint is ok
    # -gradient.T(gradient(x)) is the laplacian
    laplacian = -diff.adjoint(derivative)
    assert all_almost_equal(laplacian.asarray(),
                            [[0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0],
                             [0, 1, -4, 1, 0],
                             [0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0]])


@skip_if_no_cuda
def test_rectangle():
    # Continuous definition of problem
    space = odl.FunctionSpace(odl.Rectangle([0, 0], [1, 1]))

    # Complicated functions to check performance
    n = 5
    m = 7

    # Discretization
    d = odl.uniform_discr(space, (n, m), impl='cuda')

    fun = d.element([[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]])

    diff = ForwardDiff2D(d)
    derivative = diff(fun)

    assert all_almost_equal(derivative[0].asarray(),
                            [[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 1, -1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])

    assert all_almost_equal(derivative[1].asarray(),
                            [[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, -1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])

    # Verify that the adjoint is ok
    # -gradient.T(gradient(x)) is the laplacian
    laplacian = -diff.adjoint(derivative)
    assert all_almost_equal(laplacian.asarray(),
                            [[0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 1, -4, 1, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0]])


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\','/') + ' -v'))
