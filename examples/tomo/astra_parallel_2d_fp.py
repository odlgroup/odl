# Copyright 2014, 2015 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""ASTRA example."""

# pylint: disable=invalid-name,no-name-in-module

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import numpy as np

# Internal
import odl


def phantom(x):
    return np.float64((x[0]**2/0.2**2) + (x[1]**2/0.5**2) <= 1)


# Reconstruction domain, continuous and dcontains_set
reco_space = odl.FunctionSpace(odl.Rectangle([-1, -1], [1, 1]))
vol_shape = (50, 50)
discr_reco_space = odl.uniform_discr_fromspace(reco_space, vol_shape,
                                               dtype='float32')

# The following is still very inconvenient since you have to use the class
# constructor of Parallel2dGeometry - convenience functions to come

# Data domain, continuous and sampled
angle_range = odl.Interval(0, np.pi)
det_range = odl.Interval(-1.5, 1.5)
angles = odl.uniform_sampling(angle_range, 20)
det_pixels = odl.uniform_sampling(det_range, 50)

# Initialize the geometry
geom = odl.tomo.Parallel2dGeometry(angle_range, det_range, angles, det_pixels)

# Initialize the operator
xray_trafo = odl.tomo.DiscreteXrayTransform(discr_reco_space, geom,
                                        backend='astra')

# Make a discrete phantom
cont_phantom = reco_space.element(phantom)
discr_phantom = discr_reco_space.element(cont_phantom)

# Create data
proj_data = xray_trafo(discr_phantom)

proj_data.show()
