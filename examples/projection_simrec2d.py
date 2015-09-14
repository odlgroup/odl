# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

Copyright 2014, 2015 Holger Kohr

This file is part of ODL.

ODL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

ODL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with ODL.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import (division, print_function, unicode_literals,
                        absolute_import)
from future import standard_library
from math import sin, cos
import matplotlib.pyplot as plt

import numpy as np
import odl.operator.operator as OP
from odl.space.default import L2
import odl.space.cartesian as ds
import odl.discr.discretization as dd
from odl.set.domain import Interval, Rectangle
from odl.discr.default import DiscreteL2, l2_uniform_discretization
import SimRec2DPy as SR

standard_library.install_aliases()


class Projection(OP.LinearOperator):
    def __init__(self, volumeOrigin, voxelSize, volumeSize, detectorSize,
                 stepSize, sourcePosition, detectorOrigin, pixelDirection,
                 domain, range_):
        self.volumeOrigin = volumeOrigin
        self.voxelSize = voxelSize
        self.volumeSize = volumeSize
        self.detectorSize = detectorSize
        self.stepSize = stepSize
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection
        self.domain = domain
        self.range = range_

    def _apply(self, volume, out):
        forward = SR.SRPyForwardProject.SimpleForwardProjector(
            volume.ntuple.data.reshape(self.volumeSize), self.volumeOrigin,
            self.voxelSize, self.detectorSize, self.stepSize)

        result = forward.project(self.sourcePosition, self.detectorOrigin,
                                 self.pixelDirection)

        out[:] = result.transpose()

# Set geometry parameters
volumeSize = np.array([20.0, 20.0])
volumeOrigin = -volumeSize/2.0

detectorSize = 50.0
detectorOrigin = -detectorSize/2.0

sourceAxisDistance = 20.0
detectorAxisDistance = 20.0

# Discretization parameters
nVoxels = np.array([500, 400])
nPixels = 400
nProjection = 500

# Scale factors
voxelSize = volumeSize/nVoxels
pixelSize = detectorSize/nPixels
stepSize = voxelSize.max()/2.0

# Define projection geometries
theta = 0
x0 = np.array([cos(theta), sin(theta)])
y0 = np.array([-sin(theta), cos(theta)])

sourcePosition = -sourceAxisDistance * x0
detectorOrigin = detectorAxisDistance * x0 + detectorOrigin * y0
pixelDirection = y0 * pixelSize


dataSpace = L2(Interval(0, 1))
dataDisc = l2_uniform_discretization(dataSpace, nPixels, impl='numpy')

reconSpace = L2(Rectangle((0, 0), (1, 1)))
reconDisc = l2_uniform_discretization(reconSpace, nVoxels, impl='numpy')

# Create a phantom
phantom = SR.SRPyUtils.phantom(nVoxels)
phantomVec = reconDisc.element(phantom.flatten())

projector = Projection(volumeOrigin, voxelSize, nVoxels, nPixels, stepSize,
                       sourcePosition, detectorOrigin, pixelDirection,
                       reconDisc, dataDisc)

result = dataDisc.element()
projector.apply(phantomVec, result)

plt.plot(result.asarray())
plt.show()
