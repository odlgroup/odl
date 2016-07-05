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

"""Simple example using the operators of linearized deformations."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np
import odl


# Discrete reconstruction space: discretized functions on the rectangle
# [-1, 1]^2 with 100 samples per dimension.
discr_space = odl.uniform_discr([-1, -1], [1, 1], (100, 100), interp='linear')

# Create a template
template = odl.phantom.cuboid(discr_space, [-0.5, -0.5], [-0.25, 0])

# Define a displacement field that shifts an image horizontally by hx
# to the right and vertically by hy up. The it rotates clockwise by theta.
hx = -0.5
hy = -0.5
theta = np.pi/4
disp_func = [lambda x: (np.cos(theta) - 1) * x[0] - np.sin(theta) * x[1] + hx,
             lambda x: np.sin(theta) * x[0] + (np.cos(theta) - 1) * x[1] + hy]

# Define the deformation operator where template is fixed
fixed_templ_op = odl.deform.LinDeformFixedTempl(template)

# Discrete a displacement field space corresponding to the above
# discrete reconstruction space
disp_field_space = fixed_templ_op.domain

# Create a displacement field based on ``disp_func``
disp_field = disp_field_space.element(disp_func)

# Calculate the deformed template by the fixed template
# linearized deformation operator
deform_templ_fixed_templ = fixed_templ_op(disp_field)

# Define the derivative of the fixed template linearized
# deformation operator.
fixed_templ_deriv_op = fixed_templ_op.derivative(disp_field)

# Evaluate the derivative at the vector field that is only 1. This should be
# the same as the pointwise inner product between linearly deformed
# gradient and the said vector field
vector_field = disp_field_space.one()
fixed_templ_deriv = fixed_templ_deriv_op(vector_field)

# Evaluate the adjoint of derivative at the element that is only 1
func = discr_space.one()
fixed_templ_adj = fixed_templ_deriv_op.adjoint(func)

# Define the deformation operator where the displacement field is fixed
fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)

# Calculate the deformed template
deform_templ_fixed_disp = fixed_disp_op(template)

# Compute the adjoint of the fixed displacement
# linearized deformation operator
fixed_disp_adj = fixed_disp_op.adjoint(template)

# Show results
template.show('template')
deform_templ_fixed_templ.show('deformed template fixed template')
fixed_templ_deriv.show('derivative fixed template')
fixed_templ_adj.show('adjoint fixed template')
deform_templ_fixed_disp.show('deformed template fixed displacement')
fixed_disp_adj.show('adjoint fixed displacement')
