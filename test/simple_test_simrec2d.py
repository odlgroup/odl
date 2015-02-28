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
from math import sin,cos
import unittest

import os
import numpy as np
import SimRec2DPy as SR
import RL.operator.operatorAlternative as OP

class Projection(OP.LinearOperator):
    def __init__(self,volumeOrigin,voxelSize,detectorSize,stepSize,sourcePosition,detectorOrigin,pixelDirection):
        self.volumeOrigin = volumeOrigin
        self.voxelSize = voxelSize
        self.detectorSize = detectorSize
        self.stepSize = stepSize
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection

    def apply(self,data):
        forward = SR.SimpleForwardProjector(data,self.volumeOrigin,self.voxelSize,self.detectorSize,self.stepSize)
        return forward.project(self.sourcePosition,self.detectorOrigin,self.pixelDirection)

    def applyAdjoint(self,rhs):
        raise NotImplementedError("")

class ProjectionTest(unittest.TestCase):
    def testMultiply(self):
        side = 100
        size = np.array([side,side])
        data = SR.phantom(size)
        volumeOrigin = np.array([-10.0,-10.0])
        voxelSize = np.array([20.0/side,20.0/side])
        detectorSize = 200
        stepSize = 0.01

        theta = 0.1235
        x0 = np.array([cos(theta), sin(theta)])
        y0 = np.array([-sin(theta), cos(theta)])

        sourcePosition = -20 * x0
        detectorOrigin = 20 * x0 + -30 * y0
        pixelDirection = y0 * 60.0 / detectorSize

        projector = Projection(volumeOrigin,voxelSize,detectorSize,stepSize,sourcePosition,detectorOrigin,pixelDirection)
    
        ret = projector.apply(data)

        print (SR.printArray(ret.transpose()))

if __name__ == '__main__':
    unittest.main(exit = False)
