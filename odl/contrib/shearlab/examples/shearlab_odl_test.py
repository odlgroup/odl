
# coding: utf-8

# # <center> Shearlab_odl_test </center>

import odl
import sys
sys.path.append('../')
import shearlab_operator
import odl.contrib.pyshearlab

space = odl.uniform_discr([-1, -1], [1, 1], [512,512])

with odl.util.Timer('Shearlet System Generation shearlab'):
    op_pyshearlab = odl.contrib.pyshearlab.PyShearlabOperator(space, num_scales=4)

with odl.util.Timer('Shearlet System Generation pysherlab'):
    op_shearlab = shearlab_operator.ShearlabOperator(space, num_scales=4)

phantom = odl.phantom.shepp_logan(space, True)

y_shearlab = op_shearlab(phantom)
y_shearlab.show('Shearlet coefficients')

z_shearlab = op_shearlab.inverse(y_shearlab)
z_shearlab.show('Reconstruction')

z_shearlab = op_shearlab.adjoint(y_shearlab)
z_shearlab.show('Adjoint')

