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
from math import pi

import os
import numpy as np
import RL.operator.operatorAlternative as OP
import RL.space.functionSpaces as fs
import RL.space.defaultSpaces as ds
import RL.space.defaultDiscretizations as dd
import RL.space.set as sets
import SimRec2DPy as SR
import RL.operator.defaultSolvers as solvers

import matplotlib.pyplot as plt

class ProjectionGeometry(object):
    def __init__(self, sourcePosition, detectorOrigin, pixelDirection):
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection

class Projector(OP.LinearOperator):
    def __init__(self, volumeOrigin, voxelSize, volumeSize, detectorSize, stepSize, geometries, domain, range):
        self.volumeOrigin = volumeOrigin
        self.voxelSize = voxelSize
        self.volumeSize = volumeSize
        self.detectorSize = detectorSize
        self.stepSize = stepSize
        self.geometries = geometries
        self._domain = domain
        self._range = range

    def applyImpl(self, data, out):
        forward = SR.SRPyForwardProject.SimpleForwardProjector(data.values.reshape(self.volumeSize),self.volumeOrigin,self.voxelSize,self.detectorSize,self.stepSize)

        for i in range(len(self.geometries)):
            geo = self.geometries[i]
            result = forward.project(geo.sourcePosition,geo.detectorOrigin,geo.pixelDirection)
            out[i][:] = result.transpose()

    def applyAdjointImpl(self, projections, out):
        back = SR.SRPyReconstruction.BackProjector(self.volumeSize,self.volumeOrigin,self.voxelSize)

        for i in range(len(self.geometries)):
            geo = self.geometries[i]
            back.append(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirection, projections[i].values)

        out.values = back.finalize().flatten()

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range

class ProjectionTest(unittest.TestCase):
    def testForward(self):
        side = 100
        volumeSize = np.array([side,side])
        data = SR.SRPyUtils.phantom(volumeSize)
        volumeOrigin = np.array([-10.0,-10.0])
        voxelSize = np.array([20.0/side,20.0/side])
        detectorSize = 200
        stepSize = voxelSize.max()/2.0

        nProjection = 300
        geometries = []
        for theta in np.linspace(0,2*pi,nProjection):
            x0 = np.array([cos(theta), sin(theta)])
            y0 = np.array([-sin(theta), cos(theta)])

            sourcePosition = -20 * x0
            detectorOrigin = 20 * x0 + -30 * y0
            pixelDirection = y0 * 60.0 / detectorSize
            geometries.append(ProjectionGeometry(sourcePosition, detectorOrigin, pixelDirection))
    
        projectionSpace = fs.L2(sets.Interval(0,1))
        projectionRN = ds.EuclidianSpace(detectorSize)
        projectionDisc = dd.makeUniformDiscretization(projectionSpace, projectionRN)
        dataDisc = ds.PowerSpace(projectionDisc, nProjection)

        reconSpace = fs.L2(sets.Square((0, 0), (1, 1)))
        reconRN = ds.EuclidianSpace(side*side)
        reconDisc = dd.makePixelDiscretization(reconSpace, reconRN, side, side)

        dataVec = reconDisc.makeVector(data)
        
        projector = Projector(volumeOrigin, voxelSize, volumeSize, detectorSize, stepSize, geometries, reconDisc, dataDisc)

        ret = projector(dataVec)
        recon = projector.T(ret)
        normEst = recon.norm() / dataVec.norm()
        
        plt.figure()
        plt.imshow(recon.values.reshape(volumeSize))

        x = reconDisc.zero()
        solvers.landweber(projector, x, ret, 0.5/normEst, 50, partialResults = solvers.printStatusPartial())

        plt.figure()
        plt.imshow(x.values.reshape(volumeSize))
        
        plt.show()


if __name__ == '__main__':
    unittest.main(exit=False)
