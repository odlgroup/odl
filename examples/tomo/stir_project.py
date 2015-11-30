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

"""Example projections with stir."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import matplotlib.pyplot as plt
import numpy as np

import os.path as pth
import odl

base = pth.join(pth.join(pth.dirname(pth.abspath(__file__)), 'data'), 'stir')

projection_template = str(pth.join(base, 'small.hs'))
data_template = str(pth.join(base, 'initial.hv'))

recon_sp = odl.uniform_discr(odl.FunctionSpace(odl.Cuboid([0, 0, 0],
                                                          [1, 1, 1])),
                             [15, 64, 64])

data_sp = odl.uniform_discr(odl.FunctionSpace(odl.Cuboid([0, 0, 0],
                                                         [1, 1, 1])),
                            [37, 28, 56])

proj = odl.tomo.StirProjectorFromFile(recon_sp, data_sp, data_template, projection_template)

vol = recon_sp.one()

result = proj(vol)
result.show()

back_projected = proj.adjoint(result)
back_projected.show()