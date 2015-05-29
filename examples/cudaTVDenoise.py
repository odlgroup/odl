# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
from numpy import float64

import numpy as np
from RL.operator.operator import *
from RL.space.space import *
from RL.space.euclidean import *
from RL.space.function import *
import RL.space.cuda as CS
import RL.space.discretizations as DS
import RLcpp

import matplotlib.pyplot as plt

standard_library.install_aliases()


class ForwardDiff(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space, scale=1.0):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaRN")

        self.domain = self.range = space
        self.scale = scale

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiff(rhs.data, out.data)
        out *= self.scale

    @property
    def adjoint(self):
        return ForwardDiffAdj(self.domain, self.scale)


class ForwardDiffAdj(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space, scale=1.0):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaRN")

        self.domain = self.range = space
        self.scale = scale

    def _apply(self, rhs, out):
        RLcpp.cuda.forwardDiffAdj(rhs.data, out.data)
        out *= self.scale

    @property
    def adjoint(self):
        return ForwardDiffAdj(self.domain, self.scale)


def denoise(x0, la, mu, iterations=1):
    scale = (x0.space.n - 1.0)/(x0.space.parent.domain.end -
                                x0.space.parent.domain.begin)

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
        RLcpp.cuda.sign(d.data, sign.data)

        #
        RLcpp.cuda.abs(d.data, d.data)
        RLcpp.cuda.add_scalar(d.data, -1.0/la, d.data)
        RLcpp.cuda.max_vector_scalar(d.data, 0.0, d.data)
        d *= sign

        # b = b - diff(x) + d
        diff.apply(x, xdiff)
        b -= xdiff
        b += d

    plt.plot(x)

# Continuous definition of problem
I = Interval(0, 1)
space = L2(I)

# Complicated functions to check performance
n = 1000

# Discretization
rn = CS.CudaRN(n)
d = DS.uniform_discretization(space, rn)
x = d.points()
fun = d.element(2*((x>0.3).astype(float64) - (x>0.6).astype(float64)) + np.random.rand(n))
plt.plot(fun)

la = 0.00001
mu = 200.0
denoise(fun, la, mu, 500)

plt.show()
