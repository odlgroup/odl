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

# pylint: disable=abstract-method

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

from numpy import float64

import numpy as np
from odl.operator.operator import *
from odl.sets.space import *
from odl.sets.set import *
from odl.sets.domain import Interval
from odl.space.default import L2
from odl.space.cartesian import *
from odl.space.fspace import *
from odl.discr.l2_discr import l2_uniform_discretization
import odl.space.cuda as CS
import odlpp.odlpp_cuda as cuda

import matplotlib.pyplot as plt


class ForwardDiff(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space, scale=1.0):
        if not isinstance(space.dspace, CS.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = self.range = space
        self.scale = scale

    def _apply(self, rhs, out):
        cuda.forward_diff(rhs.ntuple.data, out.ntuple.data)
        out *= self.scale

    @property
    def adjoint(self):
        return ForwardDiffAdj(self.domain, self.scale)


class ForwardDiffAdj(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space, scale=1.0):
        if not isinstance(space.dspace, CS.CudaRn):
            raise TypeError("space must be CudaRn")

        self.domain = self.range = space
        self.scale = scale

    def _apply(self, rhs, out):
        cuda.forward_diff_adj(rhs.ntuple.data, out.ntuple.data)
        out *= self.scale

    @property
    def adjoint(self):
        return ForwardDiffAdj(self.domain, self.scale)


def denoise(x0, la, mu, iterations=1):
    scale = (x0.space.dim - 1.0)/(x0.space.uspace.domain.end[0] -
                                   x0.space.uspace.domain.begin[0])

    diff = ForwardDiff(x0.space, scale)

    ran = diff.range
    dom = diff.domain

    x = x0.copy()
    f = x0.copy()
    b = ran.zero()
    d = ran.zero()
    sign = ran.zero()
    xdiff = ran.zero()
    tmp = dom.zero()

    C1 = mu/(mu+2*la)
    C2 = la/(mu+2*la)

    for i in range(iterations):
        # x = ((f * mu + (diff.T(diff(x)) + 2*x - diff.T(d-b)) * la)/(mu+2*la))
        diff.apply(x, xdiff)
        x.lincomb(C1, f, 2*C2, x)
        xdiff -= d
        xdiff += b
        diff.adjoint.apply(xdiff, tmp)
        x.lincomb(1, x, C2, tmp)

        # d = diff(x)-b
        diff.apply(x, d)
        d -= b

        # sign = d/abs(d)
        cuda.sign(d.ntuple.data, sign.ntuple.data)

        #
        cuda.abs(d.ntuple.data, d.ntuple.data)
        cuda.add_scalar(d.ntuple.data, -1.0/la, d.ntuple.data)
        cuda.max_vector_scalar(d.ntuple.data, 0.0, d.ntuple.data)
        d *= sign

        # b = b - diff(x) + d
        diff.apply(x, xdiff)
        b -= xdiff
        b += d

    plt.plot(x)

# Continuous definition of problem
I = Interval(0, 1)
cont_space = L2(I)

# Complicated functions to check performance
n = 1000

# Discretization
d = l2_uniform_discretization(cont_space, n, impl='cuda')
x = d.grid.meshgrid()[0]
fun = d.element(2*((x>0.3).astype(float64) - (x>0.6).astype(float64)) + np.random.rand(n))
plt.plot(fun)

la = 0.00001
mu = 200.0
denoise(fun, la, mu, 500)

plt.show()
