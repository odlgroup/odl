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


import numpy as np

import RL.datamodel.gfunc as gf
import RL.builders.xray as xray
# from RL.utility.utility import InputValidationError

# Initialize a simple volume (cuboid with value 1 in a larger cube)
vol = np.zeros([100, 100, 100])
vol[25:75, 35:65, 45:55] = 1.0
voxel_size = 0.5
vol_func = gf.Gfunc(fvals=vol, spacing=voxel_size)

vol_func[:, :, 50].display()

# TODO:
# - wrap ASTRA forward and backward projections into a Projector class
#   + use parallel beam 3D CUDA forward projection
#   + create the simplest possible geometry (rotating sample)
# - define the Landweber algorithm in terms of operators
