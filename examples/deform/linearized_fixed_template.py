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

"""Example using the operator of fixed-template linearized deformation."""

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


# --- Example of LinDeformFixedTempl and its derivative,
# and the adjoint of the derivative --- #


# Define the deformation operator where template is fixed
fixed_templ_op = odl.deform.LinDeformFixedTempl(template)

# Apply the deformation operator to get the deformed template
deform_templ_fixed_templ = fixed_templ_op(disp_field)

# Define the derivative of the deformation operator
fixed_templ_deriv_op = fixed_templ_op.derivative(disp_field)

# Evaluate the derivative at the vector field that is only 1. This should be
# the same as the pointwise inner product between linearly deformed
# gradient and the said vector field
vector_field = disp_field_space.one()
fixed_templ_deriv = fixed_templ_deriv_op(vector_field)

# Evaluate the adjoint of derivative at the element that is only 1
func = discr_space.one()
fixed_templ_adj = fixed_templ_deriv_op.adjoint(func)

# Show results
deform_templ_fixed_templ.show('Deformed template')
fixed_templ_deriv.show('Operator derivative applied to one()')
fixed_templ_adj.show('Adjoint of operator derivative applied to one()')
