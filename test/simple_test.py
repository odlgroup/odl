# -*- coding: utf-8 -*-
"""
simple_test.py -- a simple test script

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

from math import pi
import numpy as np

from RL.datamodel import ugrid as ug
#from RL.datamodel import gfunc as gf
from RL.builders import xray

# from RL.utility.utility import InputValidationError

# Initialize a sample grid
sample_shape = [100, 100, 100]
sample_voxel_size = 0.5
sample_grid = ug.Ugrid(sample_shape, spacing=sample_voxel_size)

# Initialize detector grid
detector_shape = [200, 150]
detector_pixel_size = 0.4
detector_grid = ug.Ugrid(detector_shape, spacing=detector_pixel_size)

# Set tilt angles
tilt_angles = np.linspace(-pi / 2, pi / 2, 181, endpoint=True)

# Init the geometry
xray_geometry = xray.xray_ct_parallel_geom_3d(sample_grid, detector_grid,
                                              axis=2, angles=tilt_angles,
                                              rotating_sample=True)

print('initial sample system', xray_geometry.sample.coord_sys(0))
print('sample system at -90 deg', xray_geometry.sample.coord_sys(-pi / 2))
print('sample system at 90 deg', xray_geometry.sample.coord_sys(pi / 2))



#vol = np.zeros([100, 100, 100])
#vol[25:75, 35:65, 45:55] = 1.0
#voxel_size = 0.5
#vol_func = gf.Gfunc(fvals=vol, spacing=voxel_size)
#
#vol_func[:, :, 50].display()

# TODO:
# - wrap ASTRA forward and backward projections into a Projector class
#   + use parallel beam 3D CUDA forward projection
#
# - define the Landweber algorithm in terms of operators
