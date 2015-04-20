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
standard_library.install_aliases()
import unittest

import numpy as np
import RL.operator.operatorAlternative as op
import RL.operator.defaultSolvers as solvers 
import RL.space.defaultSpaces as ds
import RL.space.set as sets
import RL.space.defaultDiscretizations as dd
import RL.space.functionSpaces as fs
from testutils import RLTestCase, Timer, consume
from solverExamples import *

import matplotlib.pyplot as plt
from scipy import ndimage

class Convolution(op.LinearOperator):
    def __init__(self, kernel):
        if not isinstance(kernel.space, ds.RN):
            raise TypeError("Kernel must be RN vector")

        self.kernel = kernel.values
        self.adjkernel = kernel.values[::-1]
        self.space = kernel.space
        self.norm = float(sum(abs(self.kernel)))

    def applyImpl(self, rhs, out):
        ndimage.convolve(rhs.values, self.kernel, output=out.values, mode='wrap')

    def applyAdjointImpl(self, rhs, out):
        ndimage.convolve(rhs.values, self.adjkernel, output=out.values, mode='wrap')

    def opNorm(self):
        return self.norm
    
    @property
    def domain(self):
        return self.space
    
    @property
    def range(self):
        return self.space

def plotPartialResults(conv):
    while True:
        x = yield
        if x is None:
            raise StopIteration()
        else:
            plt.plot(conv(x)[:])

class TestsConvolutionVisually(RLTestCase):
    """ Does not do any asserts, only for manual checking
    """
    def setUp(self):
        #Continuous definition of problem
        I = sets.Interval(0, 10)
        l2 = fs.L2(I)

        #Complicated functions to check performance
        kernelL2 = l2.makeVector(lambda x: np.exp(x/2)*np.cos(x*1.172))
        rhsL2 = l2.makeVector(lambda x: x**2*np.sin(x)**2*(x > 5))

        #Discretization
        n = 500
        rn = ds.EuclidianSpace(n)
        d = dd.makeUniformDiscretization(l2, rn)
        kernel = d.makeVector(kernelL2)
        self.rhs = d.makeVector(rhsL2)

        #Create operator
        self.conv = Convolution(kernel)

        #Initial guess
        self.x0 = d.zero()

        #Dampening parameter for landweber
        self.iterations = 100
        self.omega = 1/self.conv.opNorm()**2

    def testCGN(self):
        plt.figure()

        partial = solvers.storePartial()
        solvers.conjugateGradient(self.conv, self.x0, self.rhs, self.iterations, partial)

        for result in partial.results:
            plt.plot(self.conv(result)[:])

        plt.plot(self.x0)

        plt.plot(self.rhs)
        plt.draw()
        
    def testLandweber(self):
        plt.figure()

        partial = solvers.storePartial()
        solvers.landweber(self.conv, self.x0, self.rhs, self.iterations, self.omega, partial)

        for result in partial.results:
            plt.plot(self.conv(result)[:])

        plt.plot(self.rhs)
        plt.draw()
        
    def testTimingCG(self):
        with Timer("Optimized CG"):
            solvers.conjugateGradient(self.conv, self.x0, self.rhs, self.iterations)

        with Timer("Basic CG"):
            conjugateGradientBase(self.conv, self.x0, self.rhs, self.iterations)

    def testTimingLW(self):
        with Timer("Optimized LW"):
            solvers.landweber(self.conv, self.x0, self.rhs, self.iterations, self.omega)

        with Timer("Basic LW"):
            landweberBase(self.conv, self.x0, self.rhs, self.iterations, self.omega)
       
if __name__ == '__main__':
    unittest.main(exit=False)
    plt.show()
