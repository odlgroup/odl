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

from __future__ import (division, print_function, unicode_literals,
                        absolute_import)
from future import standard_library

import RL.operator.operator as op
import RL.operator.solvers as solvers
import RL.space.euclidean as ds
import RL.space.set as sets
import RL.space.discretizations as dd
import RL.space.function as fs
from solverExamples import *

from RL.utility.testutils import Timer, consume

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

standard_library.install_aliases()


class Convolution(op.LinearOperator):
    def __init__(self, kernel, adjkernel=None):
        if not isinstance(kernel.space, ds.RN):
            raise TypeError("Kernel must be RN vector")

        self.kernel = kernel
        self.adjkernel = (adjkernel if adjkernel is not None
                          else kernel.space.element(kernel.values[::-1]))
        self.space = kernel.space
        self.norm = float(sum(abs(self.kernel.values)))

    def _apply(self, rhs, out):
        ndimage.convolve(rhs.data, self.kernel, output=out.data,
                         mode='wrap')

    @property
    def adjoint(self):
        return Convolution(self.adjkernel, self.kernel)

    def opNorm(self):
        return self.norm

    @property
    def domain(self):
        return self.space

    @property
    def range(self):
        return self.space


# Continuous definition of problem
continuousSpace = fs.L2(sets.Interval(0, 10))

# Complicated functions to check performance
continuousKernel = continuousSpace.element(lambda x: np.exp(x/2) *
                                           np.cos(x*1.172))
continuousRhs = continuousSpace.element(lambda x: x**2 *
                                        np.sin(x)**2*(x > 5))

# Discretization
rn = ds.EuclideanSpace(500)
d = dd.makeUniformDiscretization(continuousSpace, rn)
kernel = d.element(continuousKernel)
rhs = d.element(continuousRhs)

# Create operator
conv = Convolution(kernel)

# Dampening parameter for landweber
iterations = 100
omega = 1/conv.opNorm()**2

# Display partial
partial = solvers.ForEachPartial(lambda result: plt.plot(conv(result)[:]))

# Test CGN
plt.figure()
plt.plot(rhs)
solvers.conjugate_gradient(conv, d.zero(), rhs, iterations, partial)

# Landweber
plt.figure()
plt.plot(rhs)
solvers.landweber(conv, d.zero(), rhs, iterations, omega, partial)

# testTimingCG
with Timer("Optimized CG"):
    solvers.conjugate_gradient(conv, d.zero(), rhs, iterations)

with Timer("Base CG"):
    conjugate_gradientBase(conv, d.zero(), rhs, iterations)

# Landweber timing
with Timer("Optimized LW"):
    solvers.landweber(conv, d.zero(), rhs, iterations, omega)

with Timer("Basic LW"):
    landweberBase(conv, d.zero(), rhs, iterations, omega)

plt.show()
