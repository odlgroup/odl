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
import RL.operator.space as SP

class Projection(OP.LinearOperator):
    def __init__(self,volumeOrigin,voxelSize,volumeSize,detectorSize,stepSize,sourcePosition,detectorOrigin,pixelDirection):
        self.volumeOrigin = volumeOrigin
        self.voxelSize = voxelSize
        self.volumeSize = volumeSize
        self.detectorSize = detectorSize
        self.stepSize = stepSize
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection

    def apply(self,data):
        forward = SR.SimpleForwardProjector(data,self.volumeOrigin,self.voxelSize,self.detectorSize,self.stepSize)

        return forward.project(self.sourcePosition,self.detectorOrigin,self.pixelDirection)

    def applyAdjoint(self,projection):
        back = SR.FilteredBackProjection(self.volumeSize,self.volumeOrigin,self.voxelSize)
        back.append(self.sourcePosition,self.detectorOrigin,self.pixelDirection,projection)

        return back.finalize()

class ProjectionTest(unittest.TestCase):
    def testForward(self):
        side = 100
        volumeSize = np.array([side,side])
        data = SR.phantom(volumeSize)
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

        projector = Projection(volumeOrigin,voxelSize,volumeSize,detectorSize,stepSize,sourcePosition,detectorOrigin,pixelDirection)
    
        ret = projector.apply(data)

        #print (SR.printArray(ret.transpose()))

    def testBackward(self):
        side = 100
        volumeSize = np.array([side,side])
        data = SR.phantom(volumeSize)
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

        projector = Projection(volumeOrigin,voxelSize,volumeSize,detectorSize,stepSize,sourcePosition,detectorOrigin,pixelDirection)

        proj = projector.apply(data)
        ret = projector.applyAdjoint(proj)

        #print (SR.printArray(ret,True,30,30))

    def testAdjoint(self):
        side = 100
        volumeSize = np.array([side,side])
        data = SR.phantom(volumeSize)
        volumeOrigin = np.array([-10.0,-10.0])
        voxelSize = np.array([20.0/side,20.0/side])
        detectorSize = 200
        stepSize = 0.01

        theta = 0.0
        x0 = np.array([cos(theta), sin(theta)])
        y0 = np.array([-sin(theta), cos(theta)])

        sourcePosition = -20000 * x0
        detectorOrigin = 20 * x0 + -10 * y0
        pixelDirection = y0 * 20.0 / detectorSize

        projector = Projection(volumeOrigin,voxelSize,volumeSize,detectorSize,stepSize,sourcePosition,detectorOrigin,pixelDirection)

        proj = projector.apply(data)
        data2 = projector.applyAdjoint(proj)
        data2 = data2.T #bug in c++ code
        proj2 = projector.apply(data2)
        
        print (SR.printArray(data,True,30,30))
        print (SR.printArray(data2,True,30,30))
        print (SR.printArray(proj,True,30,30))
        print (SR.printArray(proj2,True,30,30))

        rn = SP.RN(detectorSize)
        #rnm = SP.RNM(side,side)


if __name__ == '__main__':
    unittest.main(exit=False)
