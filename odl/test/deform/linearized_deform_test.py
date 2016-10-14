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

import numpy as np
import pytest

import odl
from odl.deform import LinDeformFixedTempl, LinDeformFixedDisp
from odl.util.testutils import almost_equal


# Set up fixtures

@pytest.fixture(params=['float', 'complex'])
def dtype(request):
    return np.dtype(request.param)


@pytest.fixture(params=['linear', 'nearest'])
def interp(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    return request.param


@pytest.fixture
def space(request, ndim, interp, dtype, fn_impl):
    """Example space.

    Generates example spaces with various implementations, dimensions, dtypes
    and interpolations.
    """
    if dtype not in odl.FN_IMPLS[fn_impl].available_dtypes():
        pytest.skip('dtype not available for this backend')

    return odl.uniform_discr([-1] * ndim, [1] * ndim, [20] * ndim,
                             interp=interp, impl=fn_impl, dtype=dtype)


# Set up constants and helper functions


SIGMA = 0.3  # width of the gaussian
EPS = 0.25   # scale of the displacement field


def error_bound(interp):
    """Error bound varies with interpolation (larger for "worse")."""
    if interp == 'linear':
        return 0.1
    elif interp == 'nearest':
        return 0.2


def prod(x):
    """Product of a sequence."""
    prod = 1
    for xi in x:
        prod = prod * xi
    return prod


def template_function(x):
    """Gaussian function with std SIGMA."""
    return np.exp(-sum(xi ** 2 for xi in x) / SIGMA ** 2)


def template_grad_factory(n):
    """Gradient of the gaussian."""
    def template_grad_i(i):
        # Indirection for lambda capture
        return lambda x: -2 * x[i] / SIGMA ** 2 * template_function(x)
    return [template_grad_i(i) for i in range(n)]


def disp_field_factory(n):
    """Displacement field.

    In 1d: (x)
    In 2d: (xy, y)
    In 3d: (xyz, y, z)
    etc...
    """
    def coordinate_projection_i(i):
        # Indirection for lambda capture
        return lambda x: EPS * x[i]

    lst = [lambda x: EPS * prod(x)]
    lst.extend(coordinate_projection_i(i) for i in range(1, n))
    return lst


def exp_div_inv_disp(x):
    """Exponential of the divergence of the displacement field.

    In 1d: exp(- EPS)
    In 2d: exp(- EPS * (y + 1))
    In 2d: exp(- EPS * (yz + 2))
    """
    return np.exp(- EPS * (prod(x[1:]) + (len(x) - 1)))


def displaced_points(x):
    """Displaced coordinate points."""
    disp = [dsp(x) for dsp in disp_field_factory(len(x))]
    return [xi + di for xi, di in zip(x, disp)]


def deformed_template(x):
    """Deformed template."""
    return template_function(displaced_points(x))


def vector_field_factory(n):
    """Vector field for the gradient.

    In 1d: (x)
    In 2d: (x, y)
    In 3d: (x, y, z)
    etc...
    """
    def vector_field_i(i):
        return lambda x: x[i]
    return [vector_field_i(i) for i in range(n)]


def template_deformed_grad_factory(n):
    """Deformed gradient."""
    templ_grad = template_grad_factory(n)

    def template_deformed_gradi(i):
        # Indirection for lambda capture
        return lambda x: templ_grad[i](displaced_points(x))

    return [template_deformed_gradi(i) for i in range(n)]


def fixed_templ_deriv(x):
    """Derivative taken in disp_field and evaluated in vector_field."""
    dg = [tdgf(x) for tdgf in template_deformed_grad_factory(len(x))]
    v = [vff(x) for vff in vector_field_factory(len(x))]
    return sum(dgi * vi for dgi, vi in zip(dg, v))


def inv_deformed_template(x):
    """Analytic inverse deformation of the template function."""
    disp = [dsp(x) for dsp in disp_field_factory(len(x))]
    disp_x = [xi - di for xi, di in zip(x, disp)]
    return template_function(disp_x)


# --- LinDeformFixedTempl --- #


def test_fixed_templ_init():
    """Verify that the init method and checks work properly."""
    space = odl.uniform_discr(0, 1, 5)
    template = space.element(template_function)

    # Valid input
    print(LinDeformFixedTempl(template))

    # Invalid input
    with pytest.raises(TypeError):
        # template not a DiscreteLpElement
        LinDeformFixedTempl(template_function)


def test_fixed_templ_call(space):
    """Test deformation for LinDeformFixedTempl."""

    # Define the analytic template as the hat function and its gradient
    template = space.element(template_function)
    deform_op = LinDeformFixedTempl(template)

    # Calculate result and exact result
    true_deformed_templ = space.element(deformed_template)
    deformed_templ = deform_op(disp_field_factory(space.ndim))

    # Verify that the result is within error limits
    error = (true_deformed_templ - deformed_templ).norm()
    rlt_err = error / deformed_templ.norm()
    assert rlt_err < error_bound(space.interp)


def test_fixed_templ_deriv(space):
    if not space.is_rn:
        pytest.skip('derivative not implemented for complex dtypes')

    # Set up template and displacement field
    template = space.element(template_function)
    disp_field = disp_field_factory(space.ndim)
    vector_field = vector_field_factory(space.ndim)
    fixed_templ_op = LinDeformFixedTempl(template)

    # Calculate result
    fixed_templ_op_deriv = fixed_templ_op.derivative(disp_field)
    fixed_templ_deriv_comp = fixed_templ_op_deriv(vector_field)

    # Calculate the analytic result
    fixed_templ_deriv_exact = space.element(fixed_templ_deriv)

    # Verify that the result is within error limits
    error = (fixed_templ_deriv_exact - fixed_templ_deriv_comp).norm()
    rlt_err = error / fixed_templ_deriv_comp.norm()
    assert rlt_err < error_bound(space.interp)


# --- LinDeformFixedDisp --- #


def test_fixed_disp_init():
    """Verify that the init method and checks work properly."""
    space = odl.uniform_discr(0, 1, 5)
    disp_field = space.vector_field_space.element(
        disp_field_factory(space.ndim))

    # Valid input
    print(LinDeformFixedDisp(disp_field))
    print(LinDeformFixedDisp(disp_field, templ_space=space))

    # Non-valid input
    with pytest.raises(TypeError):  # displacement not ProductSpaceElement
        LinDeformFixedDisp(space.one())
    with pytest.raises(TypeError):  # templ_space not DiscreteLp
        LinDeformFixedDisp(disp_field, space.vector_field_space)
    with pytest.raises(TypeError):  # templ_space not a power space
        bad_pspace = odl.ProductSpace(space, odl.rn(3))
        LinDeformFixedDisp(disp_field, bad_pspace)
    with pytest.raises(TypeError):  # templ_space not based on DiscreteLp
        bad_pspace = odl.ProductSpace(odl.rn(2), 1)
        LinDeformFixedDisp(disp_field, bad_pspace)
    with pytest.raises(TypeError):  # wrong dtype on templ_space
        wrong_dtype = odl.ProductSpace(space.astype(complex), 1)
        LinDeformFixedDisp(disp_field, wrong_dtype)
    with pytest.raises(ValueError):  # vector field spaces don't match
        bad_space = odl.uniform_discr(0, 1, 10)
        LinDeformFixedDisp(disp_field, bad_space)


def test_fixed_disp_call(space):
    """Verify that LinDeformFixedDisp produces the correct deformation."""
    template = space.element(template_function)
    disp_field = space.vector_field_space.element(
        disp_field_factory(space.ndim))

    # Calculate result and exact result
    deform_op = LinDeformFixedDisp(disp_field, templ_space=space)
    deformed_templ = deform_op(template)
    true_deformed_templ = space.element(deformed_template)

    # Verify that the result is within error limits
    error = (true_deformed_templ - deformed_templ).norm()
    rlt_err = error / deformed_templ.norm()
    assert rlt_err < error_bound(space.interp)


def test_fixed_disp_inv(space):
    """Verify that the inverse of LinDeformFixedDisp is correct."""
    # Set up template and displacement field
    template = space.element(template_function)
    disp_field = space.vector_field_space.element(
        disp_field_factory(space.ndim))

    # Verify that the inverse is in fact a (left and right) inverse
    deform_op = LinDeformFixedDisp(disp_field, templ_space=space)

    result_op_inv = deform_op(deform_op.inverse(template))
    error = (result_op_inv - template).norm()
    rel_err = error / template.norm()
    assert rel_err < 2 * error_bound(space.interp)  # need a bit more tolerance

    result_inv_op = deform_op.inverse(deform_op(template))
    error = (result_inv_op - template).norm()
    rel_err = error / template.norm()
    assert rel_err < 2 * error_bound(space.interp)  # need a bit more tolerance


def test_fixed_disp_adj(space):
    """Verify that the adjoint of LinDeformFixedDisp is correct."""
    # Set up template and displacement field
    template = space.element(template_function)
    disp_field = space.vector_field_space.element(
        disp_field_factory(space.ndim))

    # Calculate result
    deform_op = LinDeformFixedDisp(disp_field, templ_space=space)
    deformed_templ_adj = deform_op.adjoint(template)

    # Calculate the analytic result
    true_deformed_templ_adj = space.element(inv_deformed_template)
    exp_div = space.element(exp_div_inv_disp)
    true_deformed_templ_adj *= exp_div

    # Verify that the result is within error limits
    error = (deformed_templ_adj - true_deformed_templ_adj).norm()
    rel_err = error / true_deformed_templ_adj.norm()
    assert rel_err < error_bound(space.interp)

    # Verify the adjoint definition <Ax, x> = <x, A^* x>
    deformed_templ = deform_op(template)
    inner1 = deformed_templ.inner(template)
    inner2 = template.inner(deformed_templ_adj)
    assert almost_equal(inner1, inner2, places=1)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
