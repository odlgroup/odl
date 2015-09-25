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

import odl
import odlpp.odlpp_cuda as cuda
import odlpp.odlpp_utils as utils
from odl.util.testutils import Timer

# from pooled import makePooledSpace


class RegularizationType(object):
    Anisotropic, Isotropic = range(2)


class ForwardDiff2D(odl.LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space.dspace, odl.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = space
        self.range = odl.ProductSpace(space, 2)

    def _apply(self, rhs, out):
        cuda.forward_diff_2d(
            rhs.ntuple.data, out[0].ntuple.data, out[1].ntuple.data,
            self.domain.grid.shape[0], self.domain.grid.shape[1])

    @property
    def adjoint(self):
        return ForwardDiff2DAdjoint(self.domain)


class ForwardDiff2DAdjoint(odl.LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space.dspace, odl.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = odl.ProductSpace(space, 2)
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

    for i in odl.util.ProgressRange("denoising", iterations):
        # x = ((f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la))
        diff(x, xdiff)
        xdiff += d
        xdiff -= b
        diff.adjoint(xdiff, tmp)

        x.lincomb(C1, f, 2*C2, x)
        x.lincomb(1, x, C2, tmp)

        # d = diff(x)+b
        diff(x, xdiff)
        xdiff += b
        d.assign(xdiff)

        # s = xdiff[0] = sqrt(dx^2+dy^2)
        xdiff **= 2

        for i in range(1, dimension):
            xdiff[0] += xdiff[i]

        odl.cu_ntuples.sqrt(xdiff[0].ntuple, xdiff[0].ntuple)

        # c = tmp = max(s - la^-1, 0) / s
        odl.cu_ntuples.add_scalar(xdiff[0].ntuple, -1.0/la, tmp.ntuple)
        odl.cu_ntuples.max_vector_scalar(tmp.ntuple, 0.0, tmp.ntuple)
        odl.cu_ntuples.divide_vector_vector(tmp.ntuple, xdiff[0].ntuple,
                                            tmp.ntuple)

        # d = d * c = d * max(s - la^-1, 0) / s
        for i in range(dimension):
            d[i] *= tmp

        # b = b + diff(x) - d
        diff(x, xdiff)
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

    for i in odl.util.ProgressRange("denoising", iterations):
        # x = (f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la)
        diff(x, xdiff)
        xdiff += d
        xdiff -= b
        diff.adjoint(xdiff, tmp)
        x.lincomb(C1, f, 2*C2, x)
        x.lincomb(C2, tmp, 1, x)

        # d = diff(x)+b
        diff(x, d)
        d += b

        for j in range(dimension):
            # tmp = d/abs(d)
            odl.cu_ntuples.sign(d[j].ntuple, tmp.ntuple)

            # d = sign(diff(x)+b) * max(|diff(x)+b|-la^-1,0)
            odl.cu_ntuples.abs(d[j].ntuple, d[j].ntuple)
            odl.cu_ntuples.add_scalar(d[j].ntuple, -1.0/la, d[j].ntuple)
            odl.cu_ntuples.max_vector_scalar(d[j].ntuple, 0.0, d[j].ntuple)
            d[j].multiply(d[j], tmp)

        # b = b + diff(x) - d
        diff(x, xdiff)
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

    for i in odl.util.ProgressRange("denoising", iterations):
        x = (f * mu + (diff.T(diff(x)) + 2*x + diff.T(d-b)) * la)/(mu+2*la)

        d = diff(x)+b

        for i in range(dimension):
            # tmp = d/abs(d)
            odl.cu_ntuples.sign(d[i].ntuple, tmp)

            # d = sign(diff(x)+b) * max(|diff(x)+b|-la^-1,0)
            odl.cu_ntuples.abs(d[i].ntuple, d[i].ntuple)
            odl.cu_ntuples.add_scalar(d[i].ntuple, -1.0/la, d[i].ntuple)
            odl.cu_ntuples.max_vector_scalar(d[i].ntuple, 0.0, d[i].ntuple)
            d[i] *= tmp

        b = b + diff(x) - d

    return x

# Continuous definition of problem
space = odl.L2(odl.Rectangle([0, 0], [1, 1]))

# Complicated functions to check performance
n = 2000
m = 2000

# TODO make this work again
# Underlying Rn space
# rn = CS.CudaRn(n*m)
# Example of using an vector pool to reduce allocation overhead
# rnpooled = makePooledSpace(rn, maxPoolSize=5)

# Discretize
d = odl.l2_uniform_discretization(space, (n, m), impl='cuda')

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
    result = TVdenoise2DIsotropic(fun, la, mu, 100)

# Show result
plt.figure()
p = plt.imshow(result.asarray().reshape(n, m), interpolation='nearest')
plt.set_cmap('bone')
plt.axis('off')

plt.show()
