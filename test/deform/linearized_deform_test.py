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


def hat_function(x, **kwargs):
    """Hat function with support in square with side 2r."""
    r = kwargs.pop('r', 0.5)
    retval = (r - abs(x[0])) * (abs(x[0]) < r)
    for xi in x[1:]:
        retval = retval * (r - abs(xi)) * (abs(xi) < r)
    return retval


def hat_function_grad0(x, **kwargs):
    """First component of gradient of the hat function."""
    r = kwargs.pop('r', 0.5)
    tmp = np.where(abs(x[0]) >= 1e-6, -x[0] / abs(x[0]) * (r - abs(x[1])), 0)
    return tmp * (abs(x[0]) < r) * (abs(x[1]) < r)


def hat_function_grad1(x, **kwargs):
    """2nd component of gradient of the hat function."""
    r = kwargs.pop('r', 0.5)
    tmp = np.where(abs(x[1]) >= 1e-6, -x[1] / abs(x[1]) * (r - abs(x[0])), 0)
    return tmp * (abs(x[0]) < r) * (abs(x[1]) < r)


def disp_field0(x):
    """First component of analytic displacement field (x,y) -> eps * (x+y)."""
    eps = 0.25
    return eps * (x[0] + x[1])


def disp_field1(x):
    """Second component of analytic displacement field (x,y) -> eps * 3xy."""
    eps = 0.25
    return eps * 3 * x[0] * x[1]


def deform_hat(x, **kwargs):
    """Analytic linear deformation of the hat function.

    The analytic displacement field (x,y) -> eps * (x+y, 3xy)."""
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0] + eps * (x[0] + x[1])
    x1 = x[1] + eps * 3 * x[0] * x[1]
    return (r - abs(x0)) * (abs(x0) < r) * (r - abs(x1)) * (abs(x1) < r)


def vector_field0(x):
    """First component of analytic vector field (x,y) -> x-y."""
    return x[0] - x[1]


def vector_field1(x):
    """Second component of analytic vector field (x,y) -> 4xy."""
    return 4 * x[0] * x[1]


def hat_deform_grad0(x, **kwargs):
    """First component of the deformed gradient of the hat function.

    It is the same as evaluating the gradient at (x, y) + eps * (2x+y, y+3xy).
    """
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0] + eps * (x[0] + x[1])
    x1 = x[1] + eps * 3 * x[0] * x[1]

    tmp = np.where(abs(x0) >= 1e-6, -x0 / abs(x0) * (r - abs(x1)), 0)
    return tmp * (abs(x0) < r) * (abs(x1) < r)


def hat_deform_grad1(x, **kwargs):
    """Second component of the deformed gradient of the hat function.

    It is the same as evaluating the gradient at (x, y) + eps * (2x+y, y+3xy).
    """
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0] + eps * (x[0] + x[1])
    x1 = x[1] + eps * 3 * x[0] * x[1]

    tmp = np.where(abs(x1) >= 1e-6, -x1 / abs(x1) * (r - abs(x0)), 0)
    return tmp * (abs(x0) < r) * (abs(x1) < r)


def fixed_templ_deriv_hat(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0] + eps * (x[0] + x[1])
    x1 = x[1] + eps * 3 * x[0] * x[1]

    tmp1 = np.where(abs(x0) >= 1e-6, -x0 / abs(x0) * (r - abs(x1)), 0)
    dg0 = tmp1 * (abs(x0) < r) * (abs(x1) < r)

    tmp2 = np.where(abs(x1) >= 1e-6, -x1 / abs(x1) * (r - abs(x0)), 0)
    dg1 = tmp2 * (abs(x0) < r) * (abs(x1) < r)

    v0 = x[0] - x[1]
    v1 = 4 * x[0] * x[1]
    return dg0 * v0 + dg1 * v1


def fixed_templ_adj_hat1(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0] + eps * (x[0] + x[1])
    x1 = x[1] + eps * 3 * x[0] * x[1]

    tmp1 = np.where(abs(x0) >= 1e-6, -x0 / abs(x0) * (r - abs(x1)), 0)
    dg0 = tmp1 * (abs(x0) < r) * (abs(x1) < r)

    hat = (r - abs(x[0])) * (abs(x[0]) < r) * (r - abs(x[1])) * (abs(x[1]) < r)
    return dg0 * hat


def fixed_templ_adj_hat2(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0] + eps * (x[0] + x[1])
    x1 = x[1] + eps * 3 * x[0] * x[1]

    tmp = np.where(abs(x1) >= 1e-6, -x1 / abs(x1) * (r - abs(x0)), 0)
    dg1 = tmp * (abs(x0) < r) * (abs(x1) < r)

    hat = (r - abs(x[0])) * (abs(x[0]) < r) * (r - abs(x[1])) * (abs(x[1]) < r)
    return dg1 * hat


def given_hat(x, **kwargs):
    r = kwargs.pop('r', 0.5)
    retval = (r - abs(x[0])) * (abs(x[0]) < r)
    for xi in x[1:]:
        retval = retval * (r - abs(xi)) * (abs(xi) < r)
    return retval


def inv_deform_hat(x, **kwargs):
    """Analytic inverse deformation of the hat function.

    The analytic inverse displacement field (x,y) -> - eps * (x+y, 3xy)."""
    r = kwargs.pop('r', 0.5)
    eps = 0.25
    x0 = x[0] - eps * (x[0] + x[1])
    x1 = x[1] - eps * 3 * x[0] * x[1]
    return (r - abs(x0)) * (abs(x0) < r) * (r - abs(x1)) * (abs(x1) < r)


def exp_div_inv_disp(x, **kwargs):
    eps = 0.25
    x0 = -(eps + 3 * eps * x[0])
    return np.exp(x0)


# Define the analytic template as the hat function and its gradient
discr_space = odl.uniform_discr([-1, -1], [1, 1], (30, 30), interp='linear')
template = discr_space.element(hat_function)

# Define the displacement field (x,y) -> eps * (x+y, 3xy)
grad_space = odl.ProductSpace(discr_space, 2)
disp_field = grad_space.element([disp_field0, disp_field1])


# Test deformation for LinDeformFixedTempl
def test_fixed_templ():
    deform_templ_exact = discr_space.element(deform_hat)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    deform_templ_comp_1 = fixed_templ_op(disp_field)

    tmp = (deform_templ_exact - deform_templ_comp_1).norm()

    rlt_err = tmp / deform_templ_comp_1.norm()

    assert rlt_err < 0.03


# Test derivative for LinDeformFixedTemplDeriv
# Define the vector field where the deriative of the fixed template
# operator is evaluated. This will be the vector field (x,y) -> (x-y, 4xy)
def test_fixed_templ_deriv():
    vector_field = grad_space.element([vector_field0, vector_field1])

    fixed_templ_deriv_exact = discr_space.element(fixed_templ_deriv_hat)

    fixed_templ_deriv_op = odl.deform.LinDeformFixedTemplDeriv(template,
                                                               disp_field)
    fixed_templ_deriv_comp_1 = fixed_templ_deriv_op(vector_field)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    fixed_templ_op_deriv = fixed_templ_op.derivative(disp_field)
    fixed_templ_deriv_comp_2 = fixed_templ_op_deriv(vector_field)

    assert all_equal(fixed_templ_deriv_comp_1, fixed_templ_deriv_comp_2)

    tmp = (fixed_templ_deriv_exact - fixed_templ_deriv_comp_1).norm()

    rlt_err = tmp / fixed_templ_deriv_comp_1.norm()

    assert rlt_err < 0.5


# Test adjoint for LinDeformFixedTemplDeriv, LinDeformFixedTemplDerivAdj
# Define the given template where the adjoint of the deriative
# the fixed template operator is evaluated.
# This will be the same as the hat function.
def test_fixed_templ_adj():
    fixed_templ_adj_exact = grad_space.element([fixed_templ_adj_hat1,
                                               fixed_templ_adj_hat2])

    fixed_templ_adj_op = odl.deform.LinDeformFixedTemplDerivAdj(template,
                                                                disp_field)
    given_template = discr_space.element(given_hat)
    fixed_templ_adj_comp_1 = fixed_templ_adj_op(given_template)

    fixed_templ_deriv_op = odl.deform.LinDeformFixedTemplDeriv(template,
                                                               disp_field)
    fixed_templ_adj_comp_2 = fixed_templ_deriv_op.adjoint(given_template)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    fixed_templ_op_deriv = fixed_templ_op.derivative(disp_field)
    fixed_templ_adj_comp_3 = fixed_templ_op_deriv.adjoint(given_template)

    assert all_equal(fixed_templ_adj_comp_1, fixed_templ_adj_comp_2)

    assert all_equal(fixed_templ_adj_comp_1, fixed_templ_adj_comp_3)

    tmp = (fixed_templ_adj_exact - fixed_templ_adj_comp_1).norm()

    rlt_err = tmp / fixed_templ_adj_comp_1.norm()

    assert rlt_err < 0.5


# Test deformation for LinDeformFixedDisp
# Define the fixed displacement field (x,y) -> eps * (x+y, 3xy)
# Define the analytic template to be deformed as the hat function
def test_fixed_disp():
    fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)
    deform_templ_comp_2 = fixed_disp_op(template)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    deform_templ_comp_1 = fixed_templ_op(disp_field)

    assert all_equal(deform_templ_comp_1, deform_templ_comp_2)


# Test adjoint of LinDeformFixedDisp
# Define the template as the point of the adjoint computed at.
# Define the above template as the hat function
def test_fixed_disp_adj():
    fixed_disp_adj_op = odl.deform.LinDeformFixedDispAdj(disp_field)
    fixed_disp_adj_comp_1 = fixed_disp_adj_op(template)

    fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)
    fixed_disp_adj_comp_2 = fixed_disp_op.adjoint(template)

    assert all_equal(fixed_disp_adj_comp_1, fixed_disp_adj_comp_2)

    inv_deform_templ_exact = discr_space.element(inv_deform_hat)
    exp_div = discr_space.element(exp_div_inv_disp)
    fixed_disp_adj_exact = exp_div * inv_deform_templ_exact

    tmp = (fixed_disp_adj_exact - fixed_disp_adj_comp_1).norm()

    rlt_err = tmp / fixed_disp_adj_comp_1.norm()

    assert rlt_err < 0.05

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
