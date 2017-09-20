"""Demonstration of basic ODL->pyshearlets integration functionality."""

import odl
import odl.contrib.pyshearlab

space = odl.uniform_discr([-1, -1], [1, 1], [128, 128])

op = odl.contrib.pyshearlab.PyShearlabOperator(space, scales=2)

phantom = odl.phantom.shepp_logan(space, True)

y = op(phantom)
y.show('shearlet coefficients')

z = op.inverse(y)
z.show('reconstruction')

z = op.adjoint(y)
z.show('adjoint')
