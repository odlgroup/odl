"""Demonstration of basic ODL->shearlets integration functionality."""

import odl
import odl.contrib.shearlab

space = odl.uniform_discr([-1, -1], [1, 1], [128, 128])

op = odl.contrib.shearlab.ShearlabOperator(space, num_scales=2)

phantom = odl.phantom.shepp_logan(space, True)

y = op(phantom)
y.show('Shearlet coefficients')

z = op.inverse(y)
z.show('Reconstruction')

z = op.adjoint(y)
z.show('Adjoint')
