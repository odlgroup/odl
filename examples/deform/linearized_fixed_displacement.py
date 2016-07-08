# Copyright 2014-2016 The ODL development group
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

"""Example using the operator of fixed-displacement linearized deformation."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import odl


# --- Create template and displacement field --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-1, 1]^2 with 100 samples per dimension.
discr_space = odl.uniform_discr([-1, -1], [1, 1], (100, 100), interp='linear')

# Create a product space for displacement field
disp_field_space = odl.ProductSpace(discr_space, discr_space.ndim)

# Create a template
template = odl.phantom.cuboid(discr_space, [-0.5, -0.5], [-0.25, 0])

# Define a displacement field that shifts an image by hx
# to the right and by hy upwards. Then it rotates clockwise by theta.
hx = -0.5
hy = -0.5
theta = np.pi / 4
disp_func = [lambda x: (np.cos(theta) - 1) * x[0] - np.sin(theta) * x[1] + hx,
             lambda x: np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1] + hy]

# Create a displacement field based on ``disp_func``
disp_field = disp_field_space.element(disp_func)

# Show template and displacement field
template.show('Template')
disp_field.show('Displacement field')


# --- Example of LinDeformFixedDisp and its adjoint --- #


# Define the deformation operator where the displacement field is fixed
fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)

# Apply the deformation operator to get the deformed template
deform_templ_fixed_disp = fixed_disp_op(template)

# Compute the adjoint of the operator
fixed_disp_adj = fixed_disp_op.adjoint(template)

# Show results
deform_templ_fixed_disp.show('Deformed template')
fixed_disp_adj.show('Adjoint of operator applied to template')
