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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()

# External module imports
import numpy as np
import pytest
import odl
from odl.util.testutils import (all_equal)


def hat_function(x, **kwargs):
    """Hat function with support in square with side 2r."""
    r = kwargs.pop('r', 0.5)
    retval = (r - abs(x[0])) * (abs(x[0]) < r)
    for xi in x[1:]:
        retval = retval * (r - abs(xi)) * (abs(xi) < r)
    return retval


def hat_function_grad0(x, **kwargs):
    """First component of gradient of hat function with support in square
    with side 2r."""
    r = kwargs.pop('r', 0.5)
    return (np.where(
        abs(x[0]) >= 1e-6, -x[0] / abs(x[0]) * (r - abs(x[1])), 0) *
            (abs(x[0]) < r) * (abs(x[1]) < r))


def hat_function_grad1(x, **kwargs):
    """2nd component of gradient of hat function with support in square
    with side 2r."""
    r = kwargs.pop('r', 0.5)
    return (np.where(
        abs(x[1]) >= 1e-6, -x[1] / abs(x[1]) * (r - abs(x[0])), 0) *
            (abs(x[0]) < r) * (abs(x[1]) < r))


def displacement_field0(x):
    """First component of analytic displacement field (x,y) -> eps * (x+y)."""
    eps = 0.25
    return eps*(x[0]+x[1])


def displacement_field1(x):
    """Second component of analytic displacement field (x,y) -> eps * 3xy."""
    eps = 0.25
    return eps*3*x[0]*x[1]


def deformed_hat_function(x, **kwargs):
    """Analytic deformed hat function with support in square with side r.
    The analytic displacement field (x,y) -> eps * (x+y, 3xy)."""
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0]+eps*(x[0]+x[1])
    x1 = x[1]+eps*3*x[0]*x[1]
    return (r - abs(x0)) * (abs(x0) < r) * (r - abs(x1)) * (abs(x1) < r)


def vector_field0(x):
    """First component of analytic displacement field (x,y) -> x-y."""
    return x[0]-x[1]


def vector_field1(x):
    """Second component of analytic displacement field (x,y) -> 4xy."""
    return 4*x[0]*x[1]


def hat_function_deform_grad0(x, **kwargs):
    """First component of the deformed gradient of the hat function,
    which is the same as evaluating the gradient
    at (x, y) + eps * (2x+y, y+3xy).
    """
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0]+eps*(x[0]+x[1])
    x1 = x[1]+eps*3*x[0]*x[1]

    return (np.where(abs(x0) >= 1e-6, -x0 / abs(x0) * (r - abs(x1)), 0) *
            (abs(x0) < r) * (abs(x1) < r))


def hat_function_deform_grad1(x, **kwargs):
    """Second component of the deformed gradient of the hat function,
    which is the same as evaluating the gradient
    at (x, y) + eps * (2x+y, y+3xy).
    """
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0]+eps*(x[0]+x[1])
    x1 = x[1]+eps*3*x[0]*x[1]

    return (np.where(abs(x1) >= 1e-6, -x1 / abs(x1) * (r - abs(x0)), 0) *
            (abs(x0) < r) * (abs(x1) < r))


def LinDeformFixedTemplDeriv_hat_function(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0]+eps*(x[0]+x[1])
    x1 = x[1]+eps*3*x[0]*x[1]

    dg0 = (np.where(
        abs(x0) >= 1e-6, -x0 / abs(x0) * (r - abs(x1)), 0) *
            (abs(x0) < r) * (abs(x1) < r))
    dg1 = (np.where(
        abs(x1) >= 1e-6, -x1 / abs(x1) * (r - abs(x0)), 0) *
            (abs(x0) < r) * (abs(x1) < r))
    v0 = x[0]-x[1]
    v1 = 4*x[0]*x[1]

    return dg0*v0 + dg1*v1


def LinDeformFixedTemplDerivAdj_hat_function1(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0]+eps*(x[0]+x[1])
    x1 = x[1]+eps*3*x[0]*x[1]

    dg0 = (np.where(
        abs(x0) >= 1e-6, -x0 / abs(x0) * (r - abs(x1)), 0) *
            (abs(x0) < r) * (abs(x1) < r))
    return (dg0 * (r - abs(x[0])) * (
        abs(x[0]) < r) * (r - abs(x[1])) * (abs(x[1]) < r))


def LinDeformFixedTemplDerivAdj_hat_function2(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0]+eps*(x[0]+x[1])
    x1 = x[1]+eps*3*x[0]*x[1]

    dg1 = (np.where(
        abs(x1) >= 1e-6, -x1 / abs(x1) * (r - abs(x0)), 0) *
            (abs(x0) < r) * (abs(x1) < r))
    return (dg1 * (r - abs(x[0])) * (abs(x[0]) < r) * (
        r - abs(x[1])) * (abs(x[1]) < r))


def given_hat_function(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    retval = (r - abs(x[0])) * (abs(x[0]) < r)
    for xi in x[1:]:
        retval = retval * (r - abs(xi)) * (abs(xi) < r)
    return retval


def inv_deformed_hat_function(x, **kwargs):
    """Analytic deformed hat function with support in square with side r.
    The analytic displacement field (x,y) -> - eps * (x+y, 3xy)."""
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0]-eps*(x[0]+x[1])
    x1 = x[1]-eps*3*x[0]*x[1]
    return (r - abs(x0)) * (abs(x0) < r) * (r - abs(x1)) * (abs(x1) < r)


def exp_div_inv_disp(x, **kwargs):
    eps = 0.25
    x0 = -(eps+3*eps*x[0])
    return np.exp(x0)


# Define the analytic template as the hat function and its gradient
discr_space = odl.uniform_discr([-1, -1], [1, 1], (30, 30), interp='linear')
template = discr_space.element(hat_function)

# Define the displacement field (x,y) -> eps * (x+y, 3xy)
gradient_space = odl.ProductSpace(discr_space, 2)
displacement_field = gradient_space.element(
    [displacement_field0, displacement_field1])


# Test deformation for LinDeformFixedTempl
def test_LinDeformFixedTempl():
    deformed_template_analytic = discr_space.element(deformed_hat_function)

    LinDeformFixedTempl_op = odl.deform.LinDeformFixedTempl(template)
    deformed_template_computed_1 = LinDeformFixedTempl_op(displacement_field)

    relative_error_deformed_template = \
        (deformed_template_analytic - deformed_template_computed_1).norm() / \
        deformed_template_computed_1.norm()

    assert relative_error_deformed_template < 0.03


# Test deformed gradient
def test_deformed_grad():
    template_deform_analytic_grad = gradient_space.element(
        [hat_function_deform_grad0, hat_function_deform_grad1])

    grad = odl.Gradient(template.space)
    grad_template = grad(template)
    template_deform_computed_grad = odl.deform.deform_grad(grad_template,
                                                           displacement_field)

    relative_error_deformed_gradient = \
        (template_deform_analytic_grad -
            template_deform_computed_grad).norm() / \
        template_deform_computed_grad.norm()

    assert relative_error_deformed_gradient < 0.5


# Test derivative for LinDeformFixedTemplDeriv
# Define the vector field where the deriative of the fixed template
# operator is evaluated. This will be the vector field (x,y) -> (x-y, 4xy)
def test_LinDeformFixedTemplDeriv():
    vector_field = gradient_space.element([vector_field0, vector_field1])

    LinDeformFixedTemplDeriv_analytic = discr_space.element(
        LinDeformFixedTemplDeriv_hat_function)

    LinDeformFixedTemplDeriv_op = odl.deform.LinDeformFixedTemplDeriv(
        template, displacement_field)
    LinDeformFixedTemplDeriv_computed_1 = LinDeformFixedTemplDeriv_op(
        vector_field)

    LinDeformFixedTempl_op = odl.deform.LinDeformFixedTempl(template)
    LinDeformFixedTempl_op_derivative = LinDeformFixedTempl_op.derivative(
        displacement_field)
    LinDeformFixedTemplDeriv_computed_2 = \
        LinDeformFixedTempl_op_derivative(vector_field)

    assert all_equal(LinDeformFixedTemplDeriv_computed_1,
                     LinDeformFixedTemplDeriv_computed_2)

    relative_error_LinDeforFixedTempDeriv = \
        (LinDeformFixedTemplDeriv_analytic -
            LinDeformFixedTemplDeriv_computed_1).norm() / \
        LinDeformFixedTemplDeriv_computed_1.norm()

    assert relative_error_LinDeforFixedTempDeriv < 0.5


# Test adjoint for LinDeformFixedTemplDeriv, LinDeformFixedTemplDerivAdj
# Define the given template where the adjoint of the deriative
# the fixed template operator is evaluated.
# This will be the same as the hat function.
def test_LinDeformFixedTemplDerivAdj():
    LinDeformFixedTemplDerivAdj_analytic = gradient_space.element(
        [LinDeformFixedTemplDerivAdj_hat_function1,
         LinDeformFixedTemplDerivAdj_hat_function2])

    LinDeformFixedTemplDerivAdj_op = odl.deform.LinDeformFixedTemplDerivAdj(
        template, displacement_field)
    given_template = discr_space.element(given_hat_function)
    LinDeformFixedTemplDerivAdj_computed_1 = LinDeformFixedTemplDerivAdj_op(
        given_template)

    LinDeformFixedTemplDeriv_op = odl.deform.LinDeformFixedTemplDeriv(
        template, displacement_field)
    LinDeformFixedTemplDerivAdj_computed_2 = \
        LinDeformFixedTemplDeriv_op.adjoint(given_template)

    LinDeformFixedTempl_op = odl.deform.LinDeformFixedTempl(template)
    LinDeformFixedTempl_op_derivative = LinDeformFixedTempl_op.derivative(
        displacement_field)
    LinDeformFixedTemplDerivAdj_computed_3 = \
        LinDeformFixedTempl_op_derivative.adjoint(given_template)

    assert all_equal(LinDeformFixedTemplDerivAdj_computed_1,
                     LinDeformFixedTemplDerivAdj_computed_2)

    assert all_equal(LinDeformFixedTemplDerivAdj_computed_1,
                     LinDeformFixedTemplDerivAdj_computed_3)

    relative_error_LinDeformFixedTemplDerivAdj = \
        (LinDeformFixedTemplDerivAdj_analytic -
            LinDeformFixedTemplDerivAdj_computed_1).norm() / \
        LinDeformFixedTemplDerivAdj_computed_1.norm()

    assert relative_error_LinDeformFixedTemplDerivAdj < 0.5


# Test deformation for LinDeformFixedDisp
# Define the fixed displacement field (x,y) -> eps * (x+y, 3xy)
# Define the analytic template to be deformed as the hat function
def test_LinDeformFixedDisp():
    LinDeformFixedDisp_op = odl.deform.LinDeformFixedDisp(displacement_field)
    deformed_template_computed_2 = LinDeformFixedDisp_op(template)

    LinDeformFixedTempl_op = odl.deform.LinDeformFixedTempl(template)
    deformed_template_computed_1 = LinDeformFixedTempl_op(displacement_field)

    assert all_equal(deformed_template_computed_1,
                     deformed_template_computed_2)


# Test adjoint of LinDeformFixedDisp
# Define the template as the point of the adjoint computed at.
# Define the above template as the hat function
def test_LinDeformFixedDispAdj():
    LinDeformFixedDispAdj_op = odl.deform.LinDeformFixedDispAdj(
        displacement_field)
    LinDeformFixedDispAdj_computed_1 = LinDeformFixedDispAdj_op(template)

    LinDeformFixedDisp_op = odl.deform.LinDeformFixedDisp(displacement_field)
    LinDeformFixedDispAdj_computed_2 = LinDeformFixedDisp_op.adjoint(template)

    assert all_equal(LinDeformFixedDispAdj_computed_1,
                     LinDeformFixedDispAdj_computed_2)

    inv_deformed_template_analytic = discr_space.element(
        inv_deformed_hat_function)
    exp_div = discr_space.element(exp_div_inv_disp)
    LinDeformFixedDispAdj_analytic = exp_div*inv_deformed_template_analytic

    relative_error_LinDeformFixedDispAdj = \
        (LinDeformFixedDispAdj_analytic -
            LinDeformFixedDispAdj_computed_1).norm() / \
        LinDeformFixedDispAdj_computed_1.norm()

    assert relative_error_LinDeformFixedDispAdj < 0.05

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
