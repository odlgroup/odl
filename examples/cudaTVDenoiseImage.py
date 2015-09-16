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

import numpy as np
import matplotlib.pyplot as plt

from odl.operator.operator import *
from odl.sets.space import *
from odl.sets.domain import Rectangle
from odl.sets.pspace import ProductSpace
from odl.space.cartesian import *
from odl.space.default import L2
from odl.discr.l2_discr import l2_uniform_discretization
import odl.space.cuda as CS
import odlpp.odlpp_cuda as cuda
import odlpp.odlpp_utils as utils
from odl.util.testutils import Timer

from pooled import makePooledSpace


class RegularizationType(object):
    Anisotropic, Isotropic = range(2)


class ForwardDiff2D(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space.dspace, CS.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = space
        self.range = ProductSpace(space, 2)

    def _apply(self, rhs, out):
        cuda.forward_diff_2d(
            rhs.ntuple.data, out[0].ntuple.data, out[1].ntuple.data,
            self.domain.grid.shape[0], self.domain.grid.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2DAdjoint(self.domain)


class ForwardDiff2DAdjoint(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space.dspace, CS.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = ProductSpace(space, 2)
        self.range = space

    def _apply(self, rhs, out):
        cuda.forward_diff_2d_adj(
            rhs[0].ntuple.data, rhs[1].ntuple.data, out.ntuple.data,
            self.range.grid.shape[0], self.range.grid.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2D(self.range)


def TVdenoise2DIsotropic(x0, la, mu, iterations=1):
    diff = ForwardDiff2D(x0.space)

    dimension = 2

    L2 = diff.domain
    L2xL2 = diff.range

    x = x0.copy()
    f = x0.copy()
    b = L2xL2.zero()
    d = L2xL2.zero()
    xdiff = L2xL2.zero()
    tmp = L2.zero()

    C1 = mu/(mu+2*la)
    C2 = la/(mu+2*la)

    for i in range(iterations):
        # x = ((f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la))
        diff.apply(x, xdiff)
        xdiff += d
        xdiff -= b
        diff.adjoint.apply(xdiff, tmp)

        L2.lincomb(C1, f, 2*C2, x)
        x.lincomb(C2, tmp)

        # d = diff(x)+b
        diff.apply(x, xdiff)
        xdiff += b
        d.assign(xdiff)

        # s = xdiff[0] = sqrt(dx^2+dy^2)
        for i in range(dimension):
            xdiff[i].multiply(xdiff[i], xdiff[i])

        for i in range(1, dimension):
            xdiff[0].lincomb(1, xdiff[i])

        CS.sqrt(xdiff[0], xdiff[0])

        # c = tmp = max(s - la^-1, 0) / s
        CS.add_scalar(xdiff[0], -1.0/la, tmp)
        CS.max_vector_scalar(tmp, 0.0, tmp)
        CS.divide_vector_vector(tmp, xdiff[0], tmp)

        # d = d * c = d * max(s - la^-1, 0) / s
        for i in range(dimension):
            d[i].multiply(d[i], tmp)

        # b = b + diff(x) - d
        diff.apply(x, xdiff)
        b += xdiff
        b -= d

    return x


def TVdenoise2DOpt(x0, la, mu, iterations=1):
    diff = ForwardDiff2D(x0.space)

    dimension = 2
    L2 = diff.domain
    L2xL2 = diff.range

    x = x0.copy()
    f = x0.copy()
    b = L2xL2.zero()
    d = L2xL2.zero()
    xdiff = L2xL2.zero()
    tmp = L2.zero()

    C1 = mu/(mu+2*la)
    C2 = la/(mu+2*la)
    for i in range(iterations):
        # x = (f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la)
        diff.apply(x, xdiff)
        xdiff += d
        xdiff -= b
        diff.adjoint.apply(xdiff, tmp)
        x.lincomb(C1, f, 2*C2, x)
        x.lincomb(C2, tmp, 1, x)

        # d = diff(x)+b
        diff.apply(x, d)
        d += b

        for i in range(dimension):
            # tmp = d/abs(d)
            CS.sign(d[i].ntuple, tmp.ntuple)

            # d = sign(diff(x)+b) * max(|diff(x)+b|-la^-1,0)
            CS.abs(d[i].ntuple, d[i].ntuple)
            CS.add_scalar(d[i].ntuple, -1.0/la, d[i].ntuple)
            CS.max_vector_scalar(d[i].ntuple, 0.0, d[i].ntuple)
            d[i].multiply(d[i], tmp)

        # b = b + diff(x) - d
        diff.apply(x, xdiff)
        b += xdiff
        b -= d

    return x


def TVdenoise2D(x0, la, mu, iterations=1):
    diff = ForwardDiff2D(x0.space)

    dimension = 2
    L2 = diff.domain
    L2xL2 = diff.range

    x = x0.copy()
    f = x0.copy()

    b = L2xL2.zero()
    d = L2xL2.zero()
    tmp = L2.zero()

    for i in range(iterations):
        x = (f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la)

        d = diff(x)+b

        for i in range(dimension):
            # tmp = d/abs(d)
            CS.sign(d[i], tmp)

            # d = sign(diff(x)+b) * max(|diff(x)+b|-la^-1,0)
            CS.abs(d[i], d[i])
            CS.add_scalar(d[i], -1.0/la, d[i])
            CS.max_vector_scalar(d[i], 0.0, d[i])
            d[i].multiply(d[i], tmp)

        b = b + diff(x) - d

    return x

# Continuous definition of problem
I = Rectangle([0, 0], [1, 1])
space = L2(I)

# Complicated functions to check performance
n = 2000
m = 2000

# TODO make this work again
# Underlying Rn space
# rn = CS.CudaRn(n*m)
# Example of using an vector pool to reduce allocation overhead
# rnpooled = makePooledSpace(rn, maxPoolSize=5)

# Discretize
d = l2_uniform_discretization(space, (n, m), impl='cuda')
x, y = d.grid.meshgrid()

data = utils.phantom([n, m])
data[1:-1, 1:-1] += np.random.rand(n-2, m-2) - 0.5
fun = d.element(data.flatten())

# Show input
plt.figure()
p = plt.imshow(fun.asarray().reshape(n, m), interpolation='nearest')
plt.set_cmap('bone')
plt.axis('off')

# Call denoising
la = 0.3
mu = 5.0
with Timer("denoising time"):
    result = TVdenoise2DOpt(fun, la, mu, 100)

# Show result
plt.figure()
p = plt.imshow(result.asarray().reshape(n, m), interpolation='nearest')
plt.set_cmap('bone')
plt.axis('off')

plt.show()
