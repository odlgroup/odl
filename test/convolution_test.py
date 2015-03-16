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
from __future__ import division,print_function, unicode_literals,absolute_import
from future import standard_library
standard_library.install_aliases()
import unittest
from math import pi

import numpy as np
from RL.operator.operatorAlternative import *
from RL.operator.space import *
import SimRec2DPy as SR
from testutils import RLTestCase

import matplotlib.pyplot as plt
from scipy import ndimage

class Convolution(LinearOperator):
    def __init__(self,kernel,space):
        self.kernel = kernel
        self.adjkernel = kernel[::-1]
        self.space = space

    def apply(self,rhs,out):
        ndimage.convolve(rhs,self.kernel, output=out, mode='wrap')

    def applyAdjoint(self,rhs,out):
        ndimage.convolve(rhs,self.adjkernel, output=out, mode='wrap')

    def domain(self):
        return self.space

    def range(self):
        return self.space

def landweber(operator,x0,rhs,omega=1., iterations=1):
    x = x0
    for i in range(iterations):
        x.linComb(-omega,operator.T(operator(x)-rhs))
    return x

class ConvolutionTest(RLTestCase):
    def testconv(self):
        n=100

        I=Interval(0,pi)
        space = L2(I)
        d = UniformDiscretization(space,n)
        x = d.points()

        kernel = d.makeVector(np.exp(x*2))
        conv = Convolution(kernel,d)
        omega = 1/d.integrate(abs(kernel))**2

        rhs = d.makeVector(x**2*np.sin(x)**2)

        result = landweber(conv,d.zero(),rhs,omega,100)

        plt.plot(rhs)
        plt.plot(conv(result))
        plt.show()


if __name__ == '__main__':
    unittest.main(exit = False)