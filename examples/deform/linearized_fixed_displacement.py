"""Example using the operator of fixed-displacement linearized deformation.

The linearized deformation operator with fixed displacement ``v`` maps
a given template ``I`` to the function ``x --> I(x + v(x))``.
This operator is linear (in ``I``).

This example considers a 2D case, where the displacement field ``v``
is a Gaussian in each component, with positive sign in the first and
negative sign in the second component. Note that in the deformed image,
the value at ``x`` is **taken from** the original image at ``x + v(x)``,
hence the values are moved by ``-v(x)`` when comparing deformed and
original templates.

The adjoint is based on an approximation given by the transformation
``x --> I(x - v(x))``, multiplied by an integral transformation factor
``exp(-div(v))``.
"""

import numpy as np
import odl


# --- Create template and displacement field --- #


# Template space: discretized functions on the rectangle [-1, 1]^2 with
# 100 samples per dimension.
templ_space = odl.uniform_discr([-1, -1], [1, 1], (100, 100))

# The template is a rectangle of size 1.0 x 0.5
template = odl.phantom.cuboid(templ_space, [-0.5, -0.25], [0.5, 0.25])

# Create a product space for displacement field
disp_field_space = templ_space.tangent_bundle

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


# --- Apply LinDeformFixedDisp and its adjoint --- #


# Initialize the deformation operator with fixed displacement
deform_op = odl.deform.LinDeformFixedDisp(disp_field)

# Apply the deformation operator to get the deformed template.
deformed_template = deform_op(template)

# Evaluate the adjoint at the same template.
adj_result = deform_op.adjoint(template)

# Show results
deformed_template.show('Deformed template')
adj_result.show('Adjoint applied to the template')
