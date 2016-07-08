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

"""Tests for linearized deformation operators."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

# External
import pytest
import numpy as np

# Internal
import odl
from odl.util.testutils import all_equal


SIGMA = 0.3  # width of the gaussian
EPS = 0.25   # scale of the displacement field


def template_function(x):
    """Gaussian function with std SIGMA."""
    return np.exp(-(x[0]**2 + x[1]**2) / SIGMA ** 2)


def template_grad0(x):
    """First component of gradient of the gaussian."""
    return -2 * x[0] / SIGMA**2 * np.exp(-(x[0]**2 + x[1]**2) / SIGMA ** 2)


def template_grad1(x):
    """2nd component of gradient of the gaussian."""
    return -2 * x[1] / SIGMA**2 * np.exp(-(x[0]**2 + x[1]**2) / SIGMA ** 2)


def disp_field0(x):
    """First component of analytic displacement field (x,y) -> eps * (x+y)."""
    return EPS * (x[0] + x[1])


def disp_field1(x):
    """Second component of analytic displacement field (x,y) -> eps * 3xy."""
    return EPS * 3 * x[0] * x[1]


def deform_hat(x):
    """Analytic linear deformation of the hat function.

    The analytic displacement field (x,y) -> eps * (x+y, 3xy)."""
    x0 = x[0] + disp_field0(x)
    x1 = x[1] + disp_field1(x)
    return template_function([x0, x1])


def vector_field0(x):
    """First component of analytic vector field (x,y) -> x-y."""
    return x[0] - x[1]


def vector_field1(x):
    """Second component of analytic vector field (x,y) -> 4xy."""
    return 4 * x[0] * x[1]


def template_deform_grad0(x):
    """First component of the deformed gradient of the hat function.

    It is the same as evaluating the gradient at (x, y) + eps * (2x+y, y+3xy).
    """
    x0 = x[0] + disp_field0(x)
    x1 = x[1] + disp_field1(x)
    return template_grad0([x0, x1])


def template_deform_grad1(x):
    """Second component of the deformed gradient of the hat function.

    It is the same as evaluating the gradient at (x, y) + eps * (2x+y, y+3xy).
    """
    x0 = x[0] + disp_field0(x)
    x1 = x[1] + disp_field1(x)
    return template_grad1([x0, x1])


def fixed_templ_deriv_hat(x):
    dg0 = template_deform_grad0(x)
    dg1 = template_deform_grad1(x)
    v0 = vector_field0(x)
    v1 = vector_field1(x)
    return dg0 * v0 + dg1 * v1


def fixed_templ_adj_hat0(x):
    return template_deform_grad0(x) * template_function(x)


def fixed_templ_adj_hat1(x):
    return template_deform_grad1(x) * template_function(x)


def inv_deform_hat(x):
    """Analytic inverse deformation of the hat function.

    The analytic inverse displacement field (x,y) -> - eps * (x+y, 3xy)."""
    x0 = x[0] - disp_field0(x)
    x1 = x[1] - disp_field1(x)
    return template_function([x0, x1])


def exp_div_inv_disp(x):
    return np.exp(- EPS * (1 + 3 * x[0]))


# Test deformation for LinDeformFixedTempl
def test_fixed_templ():
    # Define the analytic template as the hat function and its gradient
    discr_space = odl.uniform_discr(
        [-1, -1], [1, 1], (30, 30), interp='linear')
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element([disp_field0, disp_field1])

    deform_templ_exact = discr_space.element(deform_hat)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    deform_templ_comp_1 = fixed_templ_op(disp_field)

    tmp = (deform_templ_exact - deform_templ_comp_1).norm()

    rlt_err = tmp / deform_templ_comp_1.norm()

    assert rlt_err < 0.05


# Test derivative for LinDeformFixedTemplDeriv
# Define the vector field where the deriative of the fixed template
# operator is evaluated. This will be the vector field (x,y) -> (x-y, 4xy)
def test_fixed_templ_deriv():
    # Define the analytic template as the hat function and its gradient
    discr_space = odl.uniform_discr(
        [-1, -1], [1, 1], (30, 30), interp='linear')
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element([disp_field0, disp_field1])

    vector_field = grad_space.element([vector_field0, vector_field1])

    fixed_templ_deriv_exact = discr_space.element(fixed_templ_deriv_hat)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    fixed_templ_op_deriv = fixed_templ_op.derivative(disp_field)
    fixed_templ_deriv_comp = fixed_templ_op_deriv(vector_field)

    tmp = (fixed_templ_deriv_exact - fixed_templ_deriv_comp).norm()

    rlt_err = tmp / fixed_templ_deriv_comp.norm()

    assert rlt_err < 0.05


# Test deformation for LinDeformFixedDisp
# Define the fixed displacement field (x,y) -> eps * (x+y, 3xy)
# Define the analytic template to be deformed as the hat function
def test_fixed_disp():
    # Define the analytic template as the hat function and its gradient
    discr_space = odl.uniform_discr(
        [-1, -1], [1, 1], (30, 30), interp='linear')
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element([disp_field0, disp_field1])

    fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)
    deform_templ_comp_2 = fixed_disp_op(template)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    deform_templ_comp_1 = fixed_templ_op(disp_field)

    assert all_equal(deform_templ_comp_1, deform_templ_comp_2)


# Test adjoint of LinDeformFixedDisp
# Define the template as the point of the adjoint taken.
# Define the above template as the hat function
def test_fixed_disp_adj():
    # Define the analytic template as the hat function and its gradient
    discr_space = odl.uniform_discr(
        [-1, -1], [1, 1], (30, 30), interp='linear')
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element([disp_field0, disp_field1])

    fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)
    fixed_disp_adj_comp = fixed_disp_op.adjoint(template)

    inv_deform_templ_exact = discr_space.element(inv_deform_hat)
    exp_div = discr_space.element(exp_div_inv_disp)
    fixed_disp_adj_exact = exp_div * inv_deform_templ_exact

    tmp = (fixed_disp_adj_exact - fixed_disp_adj_comp).norm()

    rlt_err = tmp / fixed_disp_adj_comp.norm()

    assert rlt_err < 0.05

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
