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

"""Example using the operator of fixed-template linearized deformation.

The linearized deformation operator with fixed template (image) ``I`` maps
a given displacement field ``v`` to the function ``x --> I(x + v(x))``.

Here, we consider a 2D example, where the displacement field ``v``
is a Gaussian in each component, with positive sign in the first and
negative sign in the second component. Note that in the deformed image,
the value at ``x`` is **taken from** the original image at ``x + v(x)``,
hence the values are moved by ``-v(x)`` when comparing deformed and
original templates.

The derivative and its adjoint are based on the deformation of the
gradient of the template, hence we expect to see some kind of edge image
and an "edge vector field".
"""

import numpy as np
import odl


# --- Create template and displacement field --- #


# Template space: discretized functions on the rectangle [-1, 1]^2 with
# 100 samples per dimension.
templ_space = odl.uniform_discr([-1, -1], [1, 1], (100, 100), interp='linear')

# We use a rectangle as template
template = odl.phantom.cuboid(templ_space, [-0.5, -0.25], [0.5, 0.25])

# Create a product space for displacement field
disp_field_space = templ_space.vector_field_space

# Define a displacement field that bends the template a bit towards the
# upper left. We use a list of 2 functions and discretize it using the
# disp_field_space.element() method.
sigma = 0.5
disp_func = [
    lambda x: 0.4 * np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2)),
    lambda x: -0.3 * np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2))]

disp_field = disp_field_space.element(disp_func)

# Show template and displacement field
template.show('Template')
disp_field.show('Displacement field')


# --- Apply LinDeformFixedTempl, derivative and its adjoint --- #


# Initialize the deformation operator with fixed template
deform_op = odl.deform.LinDeformFixedTempl(template)

# Apply the deformation operator to get the deformed template.
deformed_template = deform_op(disp_field)

# Initialize the derivative of the deformation operator at the
# given displacement field. The result is again an operator.
deform_op_deriv = deform_op.derivative(disp_field)

# Evaluate the derivative at the vector field that has value 1 everywhere,
# i.e. the global shift by (-1, -1).
deriv_result = deform_op_deriv(disp_field_space.one())

# Evaluate the adjoint of derivative at the image that is 1 everywhere.
deriv_adj_result = deform_op_deriv.adjoint(templ_space.one())

# Show results
deformed_template.show('Deformed template')
deriv_result.show('Operator derivative applied to one()')
deriv_adj_result.show('Adjoint of the derivative applied to one()')
