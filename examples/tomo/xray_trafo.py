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

"""Example using the discrete X-ray transform operator."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library

standard_library.install_aliases()

# External
import numpy as np

# Internal
from odl import (Interval, Rectangle, uniform_discr, uniform_sampling,
                 CircularConeFlatGeometry, DiscreteXrayTransform)
# from odl.util.phantom import shepp_logan

# Discrete reconstruction space
discr_reco_space = uniform_discr([-20, -20, -20],
                                 [20, 20, 20],
                                 [300, 300, 300], dtype='float32')

# Geometry
src_rad = 1000
det_rad = 100
angle_intvl = Interval(0, 2 * np.pi)
dparams = Rectangle([-50, -50], [50, 50])
agrid = uniform_sampling(angle_intvl, 360, as_midp=False)
dgrid = uniform_sampling(dparams, [558, 558])
geom = CircularConeFlatGeometry(angle_intvl, dparams, src_rad, det_rad, agrid,
                                dgrid)

# X-ray transform
xray_trafo = DiscreteXrayTransform(discr_reco_space, geom,
                                   backend='astra_cuda')

# Domain element
discr_vol_data = discr_reco_space.one()

# Forward projection
discr_proj_data = xray_trafo(discr_vol_data)

# Back projection
discr_reco_data = xray_trafo.adjoint(discr_proj_data)

# Shows a slice of the phantom, projections, and reconstruction
import matplotlib
matplotlib.use('qt4agg')
discr_vol_data.show(indices=np.s_[:, :, 150])
discr_proj_data.show(indices=np.s_[0, :, :])
discr_reco_data.show(indices=np.s_[:, :, 150])
