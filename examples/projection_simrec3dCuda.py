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
from math import sin, cos, pi

import numpy as np
import RL.operator.operator as OP
import RL.space.function as fs
import RL.space.cuda as cs
import RL.space.product as ps
import RL.space.discretizations as dd
import RL.space.set as sets
import SimRec2DPy as SR
import RL.operator.solvers as solvers
from RL.utility.testutils import Timer

import matplotlib.pyplot as plt

class ProjectionGeometry3D(object):
    """ Geometry for a specific projection
    """
    def __init__(self, sourcePosition, detectorOrigin, pixelDirectionU, pixelDirectionV):
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirectionU = pixelDirectionU
        self.pixelDirectionV = pixelDirectionV

class CudaProjector3D(OP.LinearOperator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, stepSize, geometries, domain, range):
        self.geometries = geometries
        self.domain = domain
        self.range = range
        self.forward = SR.SRPyCuda.CudaForwardProjector3D(nVoxels, volumeOrigin, voxelSize, nPixels, stepSize)

    def _apply(self, data, out):
        #Create projector
        self.forward.setData(data.data_ptr)

        #Project all geometries
        
        for i in range(len(self.geometries)):
            geo = self.geometries[i]
            
            self.forward.project(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirectionU, geo.pixelDirectionV, out[i].data_ptr)


#Set geometry parameters
volumeSize = np.array([224.0,224.0,224.0])
volumeOrigin = np.array([-112.0,-112.0,10.0]) #-volumeSize/2.0

detectorSize = np.array([287.04, 264.94])
detectorOrigin = np.array([-143.52, 0.0])

sourceAxisDistance = 790.0
detectorAxisDistance = 210.0

#Discretization parameters
nVoxels = np.array([448, 448, 448])
nPixels = np.array([780, 720])
nProjection = 332

#Scale factors
voxelSize = np.array([0.5, 0.5, 0.3])
pixelSize = np.array([0.368, 0.368]) #detectorSize/nPixels
stepSize = voxelSize.max()

#Define projection geometries
geometries = []
for theta in np.linspace(0, pi, nProjection, endpoint=False):
    x0 = np.array([cos(theta), sin(theta), 0.0])
    y0 = np.array([-sin(theta), cos(theta), 0.0])
    z0 = np.array([0.0, 0.0, 1.0])

    projSourcePosition = -sourceAxisDistance * x0
    projPixelDirectionU = y0 * pixelSize[0]
    projPixelDirectionV = z0 * pixelSize[1]
    projDetectorOrigin = detectorAxisDistance * x0 + detectorOrigin[0] * y0 + detectorOrigin[1] * z0
    geometries.append(ProjectionGeometry3D(projSourcePosition, projDetectorOrigin, projPixelDirectionU, projPixelDirectionV))

#Define the space of one projection
projectionSpace = fs.L2(sets.Rectangle([0,0], detectorSize))
projectionRN = cs.CudaRN(nPixels.prod())

#Discretize projection space
projectionDisc = dd.uniform_discretization(projectionSpace, projectionRN, nPixels, 'F')

#Create the data space, which is the Cartesian product of the single projection spaces
dataDisc = ps.powerspace(projectionDisc, nProjection)

#Define the reconstruction space
reconSpace = fs.L2(sets.Cube([0, 0, 0], volumeSize))

#Discretize the reconstruction space
reconRN = cs.CudaRN(nVoxels.prod())
reconDisc = dd.uniform_discretization(reconSpace, reconRN, nVoxels, 'F')

#Create a phantom
phantom = SR.SRPyUtils.phantom(nVoxels[0:2])
phantom = np.repeat(phantom, nVoxels[-1]).reshape(nVoxels)
phantomVec = reconDisc.element(phantom)

#Make the operator
projector = CudaProjector3D(volumeOrigin, voxelSize, nVoxels, nPixels, stepSize, geometries, reconDisc, dataDisc)
result = projector(phantomVec)

result = dataDisc.element()
with Timer("project"):
    projector.apply(phantomVec, result)

plt.figure()
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(result[i].asarray().T, cmap='bone', origin='lower')
    plt.axis('off')

back = SR.SRPyCuda.CudaBackProjector3D(nVoxels, volumeOrigin, voxelSize, nPixels, stepSize)
geo = geometries[0]
vol = projector.domain.element()


back.backProject(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirectionU, geo.pixelDirectionV, result[0].data_ptr, vol.data_ptr)

plt.figure()
plt.imshow(vol.asarray()[:,:,200], cmap='bone')

plt.show()