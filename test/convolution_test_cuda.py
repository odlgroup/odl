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
from RL.operator.operatorAlternative import *
from RL.space.space import *
from RL.space.defaultSpaces import *
from RL.space.functionSpaces import *
import RL.space.CudaSpace as CS
import RL.space.defaultDiscretizations as DS
from RL.space.measure import *
import RLcpp
from testutils import RLTestCase, Timer, consume
from solverExamples import *

import matplotlib.pyplot as plt

class CudaConvolution(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, kernel):
        if not isinstance(kernel.space, CS.CudaRN):
            raise TypeError("Kernel must be CudaRN vector")
        
        self.space = kernel.space
        self.kernel = kernel
        self.adjkernel = self.space.makeVector(kernel[::-1]) #The adjoint is the kernel reversed
        self.norm = float(sum(abs(self.kernel[:]))) #eval at host

    def applyImpl(self, rhs, out):
        RLcpp.cuda.conv(rhs.impl, self.kernel.impl, out.impl)

    def applyAdjointImpl(self, rhs, out):
        RLcpp.cuda.conv(rhs.impl, self.adjkernel.impl, out.impl)

    def opNorm(self): #An upper limit estimate of the operator norm
        return self.norm

    @property
    def domain(self):
        return self.space
    
    @property
    def range(self):
        return self.space


class TestsCudaConvolutionVisually(RLTestCase):
    """ Does not do any asserts, only for manual checking
    """
    def setUp(self):
        #Continuous definition of problem
        I = Interval(0, 10)
        space = L2(I)

        #Complicated functions to check performance
        n = 5000
        kernelL2 = space.makeVector(lambda x: np.exp(x/2)*np.cos(x*1.172))
        rhsL2 = space.makeVector(lambda x: x**2*np.sin(x)**2*(x > 5))

        #Discretization
        rn = CS.CudaRN(n)
        d = DS.makeDefaultUniformDiscretization(space, rn)
        self.kernel = d.makeVector(kernelL2)
        self.rhs = d.makeVector(rhsL2)

        #Create operator
        self.conv = CudaConvolution(self.kernel)

        #Initial guess
        self.x0 = d.zero()

        #Dampening parameter for landweber
        self.iterations = 10
        self.omega = 1/self.conv.opNorm()**2
        
    def testCGN(self):
        plt.figure()

        for result in conjugateGradient(self.conv, self.x0, self.rhs, iterations=self.iterations):
            plt.plot(self.conv(result)[:])

        plt.plot(self.rhs)
        plt.draw()

    def testLandweber(self):
        plt.figure()

        for result in landweber(self.conv, self.x0, self.rhs, self.omega, iterations=self.iterations):
            plt.plot(self.conv(result)[:])

        plt.plot(self.rhs)
        plt.draw()
        
    def testTimingCG(self):
        with Timer("Optimized CG"):
            consume(conjugateGradient(self.conv, self.x0, self.rhs, iterations=self.iterations))

        with Timer("Basic CG"):
            consume(conjugateGradientBase(self.conv, self.x0, self.rhs, iterations=self.iterations))

    def testTimingLW(self):
        with Timer("Optimized LW"):
            consume(landweber(self.conv, self.x0, self.rhs, self.omega, iterations=self.iterations))

        with Timer("Basic LW"):
            consume(landweberBase(self.conv, self.x0, self.rhs, self.omega, iterations=self.iterations))


if __name__ == '__main__':
    unittest.main(exit=False)
    plt.show()
