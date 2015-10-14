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
from builtins import super

from numpy import float64

import numpy as np
import odl
import odlpp.odlpp_cuda as cuda

import matplotlib.pyplot as plt


class ForwardDiff(odl.Operator):
    """ Calculates the forward difference of a vector
    """

    def __init__(self, space, scale=1.0):
        if not isinstance(space.dspace, odl.CudaRn):
            raise TypeError("space must be CudaRn")

        super().__init__(space, space, linear=True)
        self.scale = scale

    def _apply(self, rhs, out):
        cuda.forward_diff(rhs.ntuple.data, out.ntuple.data)
        out *= self.scale

    @property
    def adjoint(self):
        return ForwardDiffAdj(self.domain, self.scale)


class ForwardDiffAdj(odl.Operator):
    """ Calculates the adjoint of the sforward difference of a vector
    """

    def __init__(self, space, scale=1.0):
        if not isinstance(space.dspace, odl.CudaRn):
            raise TypeError("space must be CudaRn")

        self.scale = scale
        
        super().__init__(space, space, linear=True)

    def _apply(self, rhs, out):
        cuda.forward_diff_adj(rhs.ntuple.data, out.ntuple.data)
        out *= self.scale

    @property
    def adjoint(self):
        return ForwardDiffAdj(self.domain, self.scale)


def denoise(x0, la, mu, iterations=1):
    scale = (x0.space.size - 1.0)/(x0.space.domain.end[0] -
                                   x0.space.domain.begin[0])

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

    progress = odl.util.testutils.ProgressBar("denoising", iterations)
    for i in range(iterations):
        # x = ((f * mu + (diff.T(diff(x)) + 2*x - diff.T(d-b)) * la)/(mu+2*la))
        diff(x, out=xdiff)
        x.lincomb(C1, f, 2*C2, x)
        xdiff -= d
        xdiff += b
        diff.adjoint(xdiff, out=tmp)
        x.lincomb(1, x, C2, tmp)

        # d = diff(x)-b
        diff(x, out=d)
        d -= b

        # sign = d/abs(d)
        d.ntuple.data.sign(sign.ntuple.data)

        #
        d.ntuple.data.abs(d.ntuple.data)
        cuda.add_scalar(d.ntuple.data, -1.0/la, d.ntuple.data)
        cuda.max_vector_scalar(d.ntuple.data, 0.0, d.ntuple.data)
        d *= sign

        # b = b - diff(x) + d
        diff(x, out=xdiff)
        b -= xdiff
        b += d

        progress.update()

    plt.plot(x)

# Continuous definition of problem
cont_space = odl.L2(odl.Interval(0, 1))

# Complicated functions to check performance
n = 1000

# Discretization
d = odl.l2_uniform_discretization(cont_space, n, impl='cuda')
x = d.grid.meshgrid()[0]
fun = d.element(5*np.logical_and(x>0.3, x<0.6).astype(float64) + np.random.rand(n))
plt.plot(fun)

la = 0.00001
mu = 100.0
denoise(fun, la, mu, 200)

plt.show()
