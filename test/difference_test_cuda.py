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
from RL.space.measure import *
import RLcpp
from testutils import RLTestCase, Timer, consume
from solverExamples import *

import matplotlib.pyplot as plt

class ForwardDiff(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaRN):
            raise TypeError("space must be CudaRN")
        
        self.space = space
        
    def applyImpl(self, rhs, out):
        RLcpp.cuda.forwardDiff(rhs.impl, out.impl)

    def applyAdjointImpl(self, rhs, out):
        RLcpp.cuda.forwardDiffAdj(rhs.impl, out.impl)

    @property
    def domain(self):
        return self.space
    
    @property
    def range(self):
        return self.space

class ForwardDiff2D(LinearOperator):
    """ Calculates the circular convolution of two CUDA vectors
    """

    def __init__(self, space):
        if not isinstance(space, CS.CudaPixelDiscretization):
            raise TypeError("space must be CudaPixelDiscretization")

        self._domain = space
        self._range = ProductSpace(space,space)
        
    def applyImpl(self, rhs, out):
        RLcpp.cuda.forwardDiff2D(rhs.impl, out[0].impl, out[1].impl, self.domain.cols, self.domain.rows)

    def applyAdjointImpl(self, rhs, out):
        RLcpp.cuda.forwardDiff2DAdj(rhs[0].impl, rhs[1].impl, out.impl, self.domain.cols, self.domain.rows)

    @property
    def domain(self):
        return self._domain
    
    @property
    def range(self):
        return self._range


class TestCudaForwardDifference(RLTestCase):       
    def testCGN(self):
        #Continuous definition of problem
        I = Interval(0, 1)
        space = L2(I)

        #Complicated functions to check performance
        n = 6

        #Discretization
        d = CS.CudaUniformDiscretization(space, n)
        fun = d.makeVector([1,2,5,3,2,1])

        #Create operator
        diff = ForwardDiff(d)

        self.assertAllAlmostEquals(diff(fun),[0,3,-2,-1,-1,0])
        self.assertAllAlmostEquals(diff.T(fun),[0,-1,-3,2,1,0])
        self.assertAllAlmostEquals(diff.T(diff(fun)),[0,-3,5,-1,0,0])

class TestCudaForwardDifference2D(RLTestCase):       
    def testCGN(self):
        #Continuous definition of problem
        I = Square([0,0],[1,1])
        space = L2(I)

        #Complicated functions to check performance
        n = 5
        m = 5

        #Discretization
        d = CS.CudaPixelDiscretization(space, n, m)
        x,y = d.points()
        fun = d.makeVector([[0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,1,0,0],
                            [0,0,0,0,0],
                            [0,0,0,0,0]])

        diff = ForwardDiff2D(d)
        derivative = diff(fun)
        self.assertAllAlmostEquals(derivative[0][:].reshape(n,m),[[0,0,0,0,0],
                                                                  [0,0,0,0,0],
                                                                  [0,1,-1,0,0],
                                                                  [0,0,0,0,0],
                                                                  [0,0,0,0,0]])

        self.assertAllAlmostEquals(derivative[1][:].reshape(n,m),[[0,0,0,0,0],
                                                                  [0,0,1,0,0],
                                                                  [0,0,-1,0,0],
                                                                  [0,0,0,0,0],
                                                                  [0,0,0,0,0]])


if __name__ == '__main__':
    unittest.main(exit=False)
    plt.show()
