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


dtype_params = ['float']
dtype_ids = [' dtype={} '.format(dtype) for dtype in dtype_params]


@pytest.fixture(scope="module", ids=dtype_ids, params=dtype_params)
def dtype(request):
    return request.param


interp_params = ['linear', 'nearest']
interp_ids = [' interp={} '.format(interp) for interp in interp_params]


@pytest.fixture(scope="module", ids=interp_ids, params=interp_params)
def interp(request):
    return request.param


space_params = [1, 2, 3]
space_ids = [' dimension={} '.format(impl) for impl in space_params]


@pytest.fixture(scope="module", ids=space_ids, params=space_params)
def discr_space(request, interp, fn_impl, dtype):
    ndim = request.param

    discr_space = odl.uniform_discr(
        [-1] * ndim, [1] * ndim, [30] * ndim,
        interp=interp, impl=fn_impl, dtype=dtype)

    return discr_space


SIGMA = 0.3  # width of the gaussian
EPS = 0.25   # scale of the displacement field


def error_bound(interp):
    if interp == 'linear':
        return 0.1
    elif interp == 'nearest':
        return 0.2


def template_function(x):
    """Gaussian function with std SIGMA."""
    return np.exp(-sum(xi ** 2 for xi in x) / SIGMA ** 2)


def template_grad_factory(n):
    """gradient of the gaussian."""
    def template_grad_i(i):
        return lambda x: -2 * x[i] / SIGMA**2 * template_function(x)
    return [template_grad_i(i) for i in range(n)]


def disp_field_factory(n):
    """Displacement field.

    In 1d: (x)
    In 2d: (xy, y)
    In 3d: (xyz, y, z)
    etc...
    """
    lst = [lambda x: EPS * np.prod(x)]
    lst += [(lambda i: (lambda x: EPS * x[i]))(i) for i in range(1, n)]
    return lst


def exp_div_inv_disp(x):
    prod = 1
    for xi in x:
        prod = prod * xi
    return np.exp(- EPS * (prod + (len(x) - 1)))


def displaced_points(x):
    disp = [dsp(x) for dsp in disp_field_factory(len(x))]
    return [xi + di for xi, di in zip(x, disp)]


def deform_template(x):
    return template_function(displaced_points(x))


def vector_field_factory(n):
    def vector_field_i(i):
        return lambda x: x[i]
    return [vector_field_i(i) for i in range(n)]


def template_deform_grad_factory(n):
    """First component of the deformed gradient of the hat function.

    It is the same as evaluating the gradient at (x, y) + eps * (2x+y, y+3xy).
    """
    templ_grad = template_grad_factory(n)

    def template_deform_gradi(i):
        # Indirection for lambda capture
        return lambda x: templ_grad[i](displaced_points(x))

    return [template_deform_gradi(i) for i in range(n)]


def fixed_templ_deriv(x):
    dg = [tdgf(x) for tdgf in template_deform_grad_factory(len(x))]
    v = [vff(x) for vff in vector_field_factory(len(x))]
    return sum(dgi * vi for dgi, vi in zip(dg, v))


def inv_deform_hat(x):
    """Analytic inverse deformation of the hat function."""
    disp = [dsp(x) for dsp in disp_field_factory(len(x))]
    disp_x = [xi - di for xi, di in zip(x, disp)]
    return template_function(disp_x)


# Test deformation for LinDeformFixedTempl
def test_fixed_templ(discr_space):
    # Define the analytic template as the hat function and its gradient
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element(disp_field_factory(discr_space.ndim))

    deform_templ_exact = discr_space.element(deform_template)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    deform_templ_comp_1 = fixed_templ_op(disp_field)

    tmp = (deform_templ_exact - deform_templ_comp_1).norm()

    rlt_err = tmp / deform_templ_comp_1.norm()

    assert rlt_err < error_bound(discr_space.interp)


# Test derivative for LinDeformFixedTemplDeriv
# Define the vector field where the deriative of the fixed template
# operator is evaluated. This will be the vector field (x,y) -> (x-y, 4xy)
def test_fixed_templ_deriv(discr_space):
    # Define the analytic template as the hat function and its gradient
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element(disp_field_factory(discr_space.ndim))

    vector_field = grad_space.element(vector_field_factory(discr_space.ndim))

    fixed_templ_deriv_exact = discr_space.element(fixed_templ_deriv)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    fixed_templ_op_deriv = fixed_templ_op.derivative(disp_field)
    fixed_templ_deriv_comp = fixed_templ_op_deriv(vector_field)

    tmp = (fixed_templ_deriv_exact - fixed_templ_deriv_comp).norm()

    rlt_err = tmp / fixed_templ_deriv_comp.norm()

    assert rlt_err < error_bound(discr_space.interp)


# Test deformation for LinDeformFixedDisp
# Define the fixed displacement field (x,y) -> eps * (x+y, 3xy)
# Define the analytic template to be deformed as the hat function
def test_fixed_disp(discr_space):
    # Define the analytic template as the hat function and its gradient
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element(disp_field_factory(discr_space.ndim))

    fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)
    deform_templ_comp_2 = fixed_disp_op(template)

    fixed_templ_op = odl.deform.LinDeformFixedTempl(template)
    deform_templ_comp_1 = fixed_templ_op(disp_field)

    assert all_equal(deform_templ_comp_1, deform_templ_comp_2)


# Test adjoint of LinDeformFixedDisp
# Define the template as the point of the adjoint taken.
# Define the above template as the hat function
def test_fixed_disp_adj(discr_space):
    # Define the analytic template as the hat function and its gradient
    template = discr_space.element(template_function)

    # Define the displacement field (x,y) -> eps * (x+y, 3xy)
    grad_space = odl.ProductSpace(discr_space, discr_space.ndim)
    disp_field = grad_space.element(disp_field_factory(discr_space.ndim))

    fixed_disp_op = odl.deform.LinDeformFixedDisp(disp_field)
    fixed_disp_adj_comp = fixed_disp_op.adjoint(template)

    inv_deform_templ_exact = discr_space.element(inv_deform_hat)
    exp_div = discr_space.element(exp_div_inv_disp)
    fixed_disp_adj_exact = exp_div * inv_deform_templ_exact

    tmp = (fixed_disp_adj_exact - fixed_disp_adj_comp).norm()

    rlt_err = tmp / fixed_disp_adj_comp.norm()

    assert rlt_err < error_bound(discr_space.interp)

if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/')) + ' -v')
