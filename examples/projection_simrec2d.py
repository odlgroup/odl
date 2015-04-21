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
import RL.operator.operator as OP
import RL.space.functionSpaces as fs
import RL.space.defaultSpaces as ds
import RL.space.defaultDiscretizations as dd
import RL.space.set as sets
import SimRec2DPy as SR

import matplotlib.pyplot as plt

class ProjectionGeometry(object):
    def __init__(self, sourcePosition, detectorOrigin, pixelDirection):
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection

class Projection(OP.LinearOperator):
    def __init__(self, volumeOrigin, voxelSize, volumeSize, detectorSize, stepSize, sourcePosition, detectorOrigin, pixelDirection, domain, range):
        self.volumeOrigin = volumeOrigin
        self.voxelSize = voxelSize
        self.volumeSize = volumeSize
        self.detectorSize = detectorSize
        self.stepSize = stepSize
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection
        self._domain = domain
        self._range = range

    def applyImpl(self,data, out):
        forward = SR.SRPyForwardProject.SimpleForwardProjector(data.values.reshape(volumeSize),self.volumeOrigin,self.voxelSize,self.detectorSize,self.stepSize)

        result = forward.project(self.sourcePosition,self.detectorOrigin,self.pixelDirection)

        out[:] = result.transpose()

    def applyAdjointImpl(self, projection, out):
        back = SR.SRPyReconstruction.FilteredBackProjection(self.volumeSize,self.volumeOrigin,self.voxelSize)

        back.append(self.sourcePosition,self.detectorOrigin,self.pixelDirection,projection)

        out[:] = back.finalize()[:]

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range


side = 100
volumeSize = np.array([side,side])
data = SR.SRPyUtils.phantom(volumeSize)
plt.imshow(data)
plt.figure()
volumeOrigin = np.array([-10.0,-10.0])
voxelSize = np.array([20.0/side,20.0/side])
detectorSize = 200
stepSize = 0.01

theta = 0.5235
x0 = np.array([cos(theta), sin(theta)])
y0 = np.array([-sin(theta), cos(theta)])

sourcePosition = -20 * x0
detectorOrigin = 20 * x0 + -30 * y0
pixelDirection = y0 * 60.0 / detectorSize

    
dataSpace = fs.L2(sets.Interval(0,1))
dataRN = ds.EuclidianSpace(detectorSize)
dataDisc = dd.makeUniformDiscretization(dataSpace, dataRN)

reconSpace = fs.L2(sets.Square((0, 0), (1, 1)))
reconRN = ds.EuclidianSpace(side*side)
reconDisc = dd.makePixelDiscretization(reconSpace, reconRN, side, side)

dataVec = reconDisc.makeVector(data)
        
projector = Projection(volumeOrigin, voxelSize, volumeSize, detectorSize, stepSize, sourcePosition, detectorOrigin, pixelDirection, reconDisc, dataDisc)

ret = projector(dataVec)
plt.plot(ret)
plt.show()
