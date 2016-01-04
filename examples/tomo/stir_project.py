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

"""Example projection and back-projection with stir."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import os.path as pth
import stir
import odl

# Load STIR input files with data
base = pth.join(pth.join(pth.dirname(pth.abspath(__file__)), 'data'), 'stir')

volume_file = str(pth.join(base, 'initial.hv'))
volume = stir.FloatVoxelsOnCartesianGrid.read_from_file(volume_file)

projection_file = str(pth.join(base, 'small.hs'))
proj_data_in = stir.ProjData.read_from_file(projection_file)
proj_data = stir.ProjDataInMemory(proj_data_in.get_exam_info(),
                                  proj_data_in.get_proj_data_info())

# Create ODL spaces
recon_sp = odl.uniform_discr(odl.FunctionSpace(odl.Cuboid([0, 0, 0],
                                                          [1, 1, 1])),
                             [15, 64, 64])

data_sp = odl.uniform_discr(odl.FunctionSpace(odl.Cuboid([0, 0, 0],
                                                         [1, 1, 1])),
                            [37, 28, 56])

# Make STIR projector
proj = odl.tomo.stir_bindings.ForwardProjectorByBinWrapper(recon_sp,
                                                           data_sp,
                                                           volume,
                                                           proj_data)

# Create shepp-logan phantom
vol = odl.util.shepp_logan(proj.domain, modified=True)

# Project and show
result = proj(vol)
result.show()

# Also show back-projection
back_projected = proj.adjoint(result)
back_projected.show()
